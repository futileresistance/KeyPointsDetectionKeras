from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler
from optimizers import Adam, schedule
from layers import get_loss_funcs, show_gpus
from model import thin_model
from dataloader import data_gen_train, data_gen_val
from config import logs_dir, weights_best_file, training_log, base_lr, max_iter, batch_size

show_gpus()

train_samples = data_gen_train.size()
val_samples = data_gen_val.size()
iterations_per_epoch = train_samples // batch_size
adam = Adam(lr=base_lr)

loss_funcs = get_loss_funcs(batch_size)
thin_model.compile(loss=loss_funcs, optimizer=adam, metrics=["accuracy"])

checkpoint = ModelCheckpoint(weights_best_file, monitor='loss',
                             verbose=0, save_best_only=True,
                             save_weights_only=True, mode='min', period=1)
csv_logger = CSVLogger(training_log, append=True)
tb = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True,
                 write_images=False)
lrate = LearningRateScheduler(schedule)
callbacks_list = [checkpoint, csv_logger, tb, lrate]


thin_model.fit_generator(data_gen_train,
                    steps_per_epoch=train_samples // batch_size,
                    epochs=max_iter,
                    validation_data=data_gen_val,
                        validation_steps=val_samples // batch_size,
                        verbose=1, callbacks=callbacks_list, shuffle=False)