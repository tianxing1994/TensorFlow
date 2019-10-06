"""
相关函数:
tf.FIFOQueue
"""
import tensorflow as tf


def demo1():
    with tf.Session() as sess:
        q = tf.FIFOQueue(3, "float")
        init = q.enqueue_many(([0.1, 0.2, 0.3],))
        init2 = q.dequeue()
        init3 = q.enqueue(1.)

        sess.run(init)
        sess.run(init2)
        sess.run(init3)

        quelen = sess.run(q.size())
        for i in range(quelen):
            print(sess.run(q.dequeue()))
    return


def demo2():
    """
    在这个函数中, qr 创建了一个队列管理器 QueueRunner, 它调用了 2 个线程去完成此项任务.
    create_threads 函数用于启动线程, 此时线程已经开始运行.
    但是, 由于在 10 次循环结束后, 主线程结束并关闭会话.
    所以还未结束的入队列程会报错, 因为它正在调用一个已被关闭的线程.
    tensorflow.python.framework.errors_impl.CancelledError: Enqueue operation was cancelled
    这个演示, 会报错, 可改为 demo3 的方式.
    :return:
    """
    with tf.Session() as sess:
        q = tf.FIFOQueue(1000, "float32")
        counter = tf.Variable(0.0)
        add_op = tf.assign_add(counter, tf.constant(1.0))
        enqueueData_op = q.enqueue(counter)

        qr = tf.train.QueueRunner(q, enqueue_ops=[add_op, enqueueData_op] * 2)
        sess.run(tf.global_variables_initializer())
        enqueue_threads = qr.create_threads(sess, start=True)

        for i in range(10):
            print(sess.run(q.dequeue()))
    return


def demo3():
    """
    demo3 是 demo2 的另一种书写方式. 该方法不会报错. 但程序也不会自动结束.
    因为在主程序结束时, 它并不会去关闭会话, 所以入队线程可以一直执行, 最后阻塞.
    :return:
    """
    q = tf.FIFOQueue(1000, "float32")
    counter = tf.Variable(0.0)
    add_op = tf.assign_add(counter, tf.constant(1.0))
    enqueueData_op = q.enqueue(counter)

    sess = tf.Session()
    qr = tf.train.QueueRunner(q, enqueue_ops=[add_op, enqueueData_op] * 2)
    sess.run(tf.global_variables_initializer())
    enqueue_threads = qr.create_threads(sess, start=True)

    for i in range(10):
        print(sess.run(q.dequeue()))
    return


if __name__ == '__main__':
    demo1()