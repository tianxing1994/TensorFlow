# coding=utf-8
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
tf.app.flags.DEFINE_integer("data", 10, "")
tf.app.flags.DEFINE_boolean("istrain", "True", "")


def main(executable):
    print(executable)
    print(FLAGS.data)
    print(FLAGS.istrain)
    return


if __name__ == "__main__":
    tf.app.run()
