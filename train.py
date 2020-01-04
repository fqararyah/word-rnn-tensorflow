from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

from graphviz import Digraph
from graphviz import Source
import json
from tensorflow.python.client import timeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--input_encoding', type=str, default=None,
                       help='character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=4096,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=8,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=25,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--gpu_mem', type=float, default=0.95,
                       help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)

def profile(run_metadata, epoch=0):
            with open('profs/timeline_step' + str(epoch) + '.json', 'w') as f:
                # Create the Timeline object, and write it to a json file
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                f.write(chrome_trace)

def graph_to_dot(graph):
    dot = Digraph()
    for n in graph.as_graph_def().node:
        dot.node(n.name, label= n.name)
        for i in n.input:
            dot.edge(i, n.name)
    return dot
            
def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.input_encoding)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = cPickle.load(f)
        assert saved_words==data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

    model = Model(args)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_dir)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess: # fareed gpu_options=gpu_options)) as sess:
        train_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        
        #fareed
        dot_rep = graph_to_dot(sess.graph)
            #s = Source(dot_rep, filename="test.gv", format="PNG")
        with open('./profs/rnn.dot', 'w') as fwr:
            fwr.write(str(dot_rep))
    
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        
        operations_tensors = {}
        operations_names = tf.get_default_graph().get_operations()
        count1 = 0
        count2 = 0

        for operation in operations_names:
            operation_name = operation.name
            operations_info = tf.get_default_graph().get_operation_by_name(operation_name).values()
            if len(operations_info) > 0:
                if not (operations_info[0].shape.ndims is None):
                    operation_shape = operations_info[0].shape.as_list()
                    operation_dtype_size = operations_info[0].dtype.size
                    if not (operation_dtype_size is None):
                        operation_no_of_elements = 1
                        for dim in operation_shape:
                            if not(dim is None):
                                operation_no_of_elements = operation_no_of_elements * dim
                        total_size = operation_no_of_elements * operation_dtype_size
                        operations_tensors[operation_name] = total_size
                    else:
                        count1 = count1 + 1
                else:
                    count1 = count1 + 1
                    operations_tensors[operation_name] = -1

                #   print('no shape_1: ' + operation_name)
                #  print('no shape_2: ' + str(operations_info))
                #  operation_namee = operation_name + ':0'
                # tensor = tf.get_default_graph().get_tensor_by_name(operation_namee)
                # print('no shape_3:' + str(tf.shape(tensor)))
                # print('no shape:' + str(tensor.get_shape()))

            else:
                # print('no info :' + operation_name)
                # operation_namee = operation.name + ':0'
                count2 = count2 + 1
                operations_tensors[operation_name] = -1

                # try:
                #   tensor = tf.get_default_graph().get_tensor_by_name(operation_namee)
                # print(tensor)
                # print(tf.shape(tensor))
                # except:
                # print('no tensor: ' + operation_namee)
        print(count1)
        print(count2)
    
        with open('./profs/tensors_sz_32.txt', 'w') as f:
            for tensor, size in operations_tensors.items():
                f.write('"' + tensor + '"::' + str(size) + '\n') 
        #end fareed
    
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(model.epoch_pointer.eval(), args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            speed = 0
            if args.init_from is None:
                assign_op = model.epoch_pointer.assign(e)
                sess.run(assign_op)
            if args.init_from is not None:
                data_loader.pointer = model.batch_pointer.eval()
                args.init_from = None
            for b in range(data_loader.pointer, data_loader.num_batches):
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                start = time.time()                
                
                if b% 10 == 7:
                    summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed, run_metadata=run_metadata, options=options)
                    profile(run_metadata, b)
                    if b == 7:
                        options_mem = tf.profiler.ProfileOptionBuilder.time_and_memory()
                        options_mem["min_bytes"] = 0
                        options_mem["min_micros"] = 0
                        options_mem["output"] = 'file:outfile=./profs/mem.txt'
                        options_mem["select"] = ("bytes", "peak_bytes", "output_bytes",
                                             "residual_bytes")
                        mem = tf.profiler.profile(tf.get_default_graph(), run_meta=run_metadata, cmd="scope", options=options_mem)

                else:
                    summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed)
                    speed = time.time() - start              
                    train_writer.add_summary(summary, e * data_loader.num_batches + b)

                if (e * data_loader.num_batches + b) % int(args.batch_size /10) == 0 and b % 10 != 7:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, speed))
                """ if (e * data_loader.num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path)) """
        train_writer.close()

if __name__ == '__main__':
    main()
