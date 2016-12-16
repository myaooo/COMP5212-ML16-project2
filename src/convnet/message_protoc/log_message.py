from . import log_message_pb2 as message


def create_training_log_message(epoch, batch, batch_num, batch_loss, learning_rate, time, **kwargs):
    msg = message.TrainLog()
    msg.epoch = epoch
    msg.batch = batch
    msg.batch_num = batch_num
    msg.batch_loss = batch_loss
    msg.learng_rate = learning_rate
    msg.time = time

    if len(kwargs) != 0:
        msg.eval_message = create_evaluation_log_message(**kwargs)
    return msg


def create_evaluation_log_message(loss, acc, acc5, time, eval_num):
    eval_msg = message.EvaluationMessage()
    eval_msg.loss = loss
    eval_msg.acc = acc
    eval_msg.acc5 = acc5
    eval_msg.time = time
    eval_msg.eval_num = eval_num
    return eval_msg


def log_beautiful_print(train_message):
    out = '[Epoch {:>3}]'.format(train_message.epoch)
    out += '(batch {:>4}/{:>4}):'.format(train_message.batch,train_message.batch_num)
    out += 'Time: {:.3}s, Batch Loss: {:.4}, lr: {:f}'.format(train_message.time,
                                                             train_message.batch_loss,
                                                             train_message.learning_rate)
    print(out)
    if train_message.hasField('eval_message'):
        eval_msg = train_message.eval_message
        temp = 'Epoch {:>3}'.format(train_message.epoch)
        out2 = '{:*^30}\n'.format('Summary: '+temp)
        out2 += 'Time: {:.3}s, Loss: {:.4}, Acc: {:.2%}, Acc5: {:.2%}, eval_num: {:d}'.format(eval_msg.time,
                                                                                              eval_msg.loss,
                                                                                              eval_msg.acc,
                                                                                              eval_msg.acc5,
                                                                                              eval_msg.eval_num)

