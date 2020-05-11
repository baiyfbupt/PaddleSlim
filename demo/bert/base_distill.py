from itertools import izip
import paddle.fluid as fluid
from paddleslim.teachers.bert.reader.cls import *
from paddleslim.nas.darts.search_space import AdaBERTClassifier

import logging
from paddleslim.common import AvgrageMeter, get_logger
logger = get_logger(__name__, level=logging.INFO)


def main():
    place = fluid.CUDAPlace(0)

    BERT_BASE_PATH = "./data/pretrained_models/uncased_L-12_H-768_A-12"
    bert_config_path = BERT_BASE_PATH + "/bert_config.json"
    vocab_path = BERT_BASE_PATH + "/vocab.txt"
    data_dir = "./data/glue_data/MNLI/"
    teacher_model_dir="./teacher_model/steps_23000.pdparams"
    num_imgs = 392702
    max_seq_len = 512
    do_lower_case = True
    batch_size = 32
    epoch = 30

    processor = MnliProcessor(
        data_dir=data_dir,
        vocab_path=vocab_path,
        max_seq_len=max_seq_len,
        do_lower_case=do_lower_case,
        in_tokens=False)

    train_reader = processor.data_generator(
        batch_size=batch_size,
        phase='train',
        epoch=epoch,
        dev_count=1,
        shuffle=True)

    val_reader = processor.data_generator(
        batch_size=batch_size,
        phase='train',
        epoch=epoch,
        dev_count=1,
        shuffle=True)



    with fluid.dygraph.guard(place):
        model = AdaBERTClassifier(
            3,
            teacher_model=teacher_model_dir,
            data_dir=data_dir
        )

        def train_one_epoch(model, train_loader, valid_loader, optimizer, epoch):
            objs = AvgrageMeter()
            ce_losses = AvgrageMeter()
            kd_losses = AvgrageMeter()
            e_losses = AvgrageMeter()
            model.train()

            step_id = 0
            for train_data, valid_data in izip(train_loader(), valid_loader()):
                loss, ce_loss, kd_loss, e_loss = model.loss(train_data)
                loss.backward()

                #NOTE grad clip is removed
                optimizer.minimize(loss)
                model.clear_gradients()

                batch_size = train_data[0].shape[0]
                objs.update(loss.numpy(), batch_size)
                ce_losses.update(ce_loss.numpy(), batch_size)
                kd_losses.update(kd_loss.numpy(), batch_size)
                e_losses.update(e_loss.numpy(), batch_size)

                if step_id % 10 == 0:
                    logger.info(
                        "Train Epoch {}, Step {}, loss {}; ce: {}; kd: {}; e: {}".
                        format(epoch, step_id,
                            loss.numpy(),
                            ce_loss.numpy(), kd_loss.numpy(), e_loss.numpy()))
                step_id += 1

        model_parameters = [
            p for p in model.parameters()
            if p.name not in [a.name for a in model.arch_parameters()]
        ]
        step_per_epoch = int(num_imgs * 0.5 / batch_size)
        learning_rate = fluid.dygraph.CosineDecay(
            0.025, step_per_epoch, epoch)
        optimizer = fluid.optimizer.MomentumOptimizer(
            learning_rate,
            0.9,
            regularization=fluid.regularizer.L2DecayRegularizer(3e-4),
            parameter_list=model_parameters)

        train_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True)

        train_loader.set_batch_generator(train_reader, places=place)
        valid_loader.set_batch_generator(val_reader, places=place)

        for epoch_id in range(epoch):
            logger.info('Epoch {}, lr {:.6f}'.format(
                epoch_id, optimizer.current_step_lr()))

            train_one_epoch(model, train_loader, valid_loader, optimizer, epoch_id)



if __name__ == '__main__':
    main()
