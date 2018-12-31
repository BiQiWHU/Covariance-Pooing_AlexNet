
# coding: utf-8

# In[1]:


from network import alex_net
from tfdata import *
import numpy as np


# In[2]:


def loss(logits, targets):
    # Get rid of extra dimensions and cast targets into integers
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Calculate cross entropy from logits and targets
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    # Take the average loss across batch size
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


# In[3]:


# Train step
def train(loss_value, model_learning_rate):
    # Create optimizer
    # my_optimizer = tf.train.MomentumOptimizer(model_learning_rate, momentum=0.9)
    my_optimizer = tf.train.AdamOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)
    return train_step


# In[4]:


# Accuracy function
def accuracy_of_batch(logits, targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions, targets)
    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy


# In[5]:


def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding="bytes").item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    get_var = tf.get_variable(subkey).assign(data)
                    session.run(get_var)


# In[6]:


def main():
    # Dataset path
    train_tfrecords = 'train.tfrecords'
    val_tfrecords='val.tfrecords'
    test_tfrecords = 'test.tfrecords'

    # Learning params  原来imagenet的学习率是0.001
    learning_rate = 0.0001
    training_iters = 2000  #    一个epoch两千次
    batch_size = 50
    test_size=420

    # Load batch
    train_img, train_label = input_pipeline(train_tfrecords, batch_size)
    val_img, val_label = input_pipeline(val_tfrecords, batch_size)
    test_img, test_label = input_pipeline(test_tfrecords, test_size)

    # Model
    with tf.variable_scope('model_definition') as scope:
        train_output = alex_net(train_img,train=True)
        scope.reuse_variables()
        val_output = alex_net(val_img, train=False)
        test_output = alex_net(test_img,train=False)

    # Loss and optimizer
    loss_op = loss(train_output, train_label)
    tf.summary.scalar('loss', loss_op)
    train_op = train(loss_op, learning_rate)

    # Evaluation
    train_accuracy = accuracy_of_batch(train_output, train_label)
    tf.summary.scalar("train accuracy", train_accuracy)

    val_accuracy = accuracy_of_batch(val_output, val_label)
    tf.summary.scalar("val accuracy", val_accuracy)

    test_accuracy = accuracy_of_batch(test_output, test_label)
    tf.summary.scalar("test accuracy", test_accuracy)

    # Init
    init = tf.global_variables_initializer()

    # Summary
    merged_summary_op = tf.summary.merge_all()

    # Create Saver
    saver = tf.train.Saver(tf.trainable_variables())

    # Launch the graph
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)
        with tf.variable_scope('model_definition'):
             load_with_skip('bvlc_alexnet.npy', sess, ['fc8'])

        summary_writer = tf.summary.FileWriter('logs', sess.graph)

        print('Start training')
        # coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess)
        for step in range(training_iters):
            step += 1
            _, loss_value = sess.run([train_op, loss_op])
            print('Generation {}: Loss = {:.5f}'.format(step, loss_value))

            # Display testing status
            if step % 50 == 0:
                acc1 = sess.run(train_accuracy)
                print(' --- Train Accuracy = {:.2f}%.'.format(100. * acc1))
                acc2 = sess.run(val_accuracy)
                print(' --- Validation Accuracy = {:.2f}%.'.format(100. * acc2))

            if step % 50 == 0:
                summary_str = sess.run(merged_summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step % 2000 == 0:
                saver.save(sess, 'checkpoint/my-model.ckpt', global_step=step)

        print("Finish Training and validation!")

        #### prediction at one time
        acc3 = sess.run(test_accuracy)
        print(' --- Test Accuracy = {:.2f}%.'.format(100. * acc3))
        # coord.request_stop()
        # coord.join(threads)


# In[7]:


if __name__ == '__main__':
    main()

