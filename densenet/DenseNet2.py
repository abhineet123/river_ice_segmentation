import tensorflow as tf

class DenseNet2:
    def __init__(self, n_layers, height, width, ch, nclass, psi_act_type=0, ksiz=3, nfilters=16, loss_type=0):

        print('Creating densenet with {} layers'.format(n_layers))

        self.lr = tf.placeholder(tf.float32, shape=None, name="lr")
        self.hp = tf.placeholder(tf.float32, shape=None, name="hp")

        self.X = tf.placeholder(tf.float32, shape=(1, None, None, ch), name="X")
        self.Y = tf.placeholder(tf.float32, shape=(height * width, nclass), name="Y")

        # height = tf.placeholder(tf.int32, shape=(), name="height")
        # width = tf.placeholder(tf.int32, shape=(), name="width")

        # self.height = height
        # self.width = width

        # weight initializer
        he_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)

        hidden_layer = tf.layers.conv2d(self.X, filters=nfilters, kernel_size=(ksiz, ksiz), padding="same",
                                        activation=tf.nn.elu, kernel_initializer=he_init)
        all_layers = [self.X, hidden_layer]

        for layer_id in range(n_layers):
            # densenet type connections
            hidden_layer = tf.layers.conv2d(tf.concat(all_layers, axis=3), filters=nfilters, kernel_size=(ksiz, ksiz),
                                       padding="same",
                                       activation=tf.nn.elu, kernel_initializer=he_init)
            all_layers.append(hidden_layer)

        if psi_act_type == 0:
            psi_act = tf.exp
            self.psi_act_name = 'exp'
        else:
            psi_act = tf.nn.elu
            self.psi_act_name = 'elu'

        psi = tf.reshape(tf.layers.conv2d(
            tf.concat(all_layers, axis=3),
            filters=nclass, kernel_size=(ksiz, ksiz), padding="same", activation=psi_act, kernel_initializer=he_init),
            [height * width, nclass])

        # normalize features to unit vectors
        phi_den = tf.tile(tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(psi), axis=1)), [height * width, 1]),
                          [1, nclass])
        phi = psi / phi_den

        self.psi = psi
        self.phi = phi
        self.phi_den = phi_den

        self.class_indices = []
        class_gather = []
        for i in range(nclass):
            class_idx = tf.placeholder(tf.int32, shape=(None, 1), name="class_idx_{:d}".format(i))
            self.class_indices.append(class_idx)
            class_gather.append(tf.gather_nd(self.phi, class_idx))

        # loss: within class and between class

        if loss_type == 0:
            self.loss_convnet = 0
            for i in range(nclass):
                loss = tf.reduce_mean(
                    tf.square(tf.matmul(class_gather[i], tf.transpose(class_gather[i])) - 1.0))
                self.loss_convnet += loss
                for j in range(i + 1, nclass):
                    loss = tf.reduce_mean(
                        tf.square(tf.matmul(class_gather[i], tf.transpose(class_gather[j]))))
                    self.loss_convnet += loss
        elif loss_type == 1:
            if nclass != 3:
                raise AssertionError('loss_type 1 can only be used with 3 classes')
            sky_indices = self.class_indices[0]
            grass_indices = self.class_indices[1]
            horse_indices = self.class_indices[2]
            loss_00 = tf.reduce_mean(
                tf.square(
                    tf.matmul(tf.gather_nd(phi, sky_indices), tf.transpose(tf.gather_nd(phi, sky_indices))) - 1.0))
            loss_11 = tf.reduce_mean(
                tf.square(
                    tf.matmul(tf.gather_nd(phi, grass_indices), tf.transpose(tf.gather_nd(phi, grass_indices))) - 1.0))
            loss_22 = tf.reduce_mean(
                tf.square(
                    tf.matmul(tf.gather_nd(phi, horse_indices), tf.transpose(tf.gather_nd(phi, horse_indices))) - 1.0))
            loss_01 = tf.reduce_mean(
                tf.square(tf.matmul(tf.gather_nd(phi, sky_indices), tf.transpose(tf.gather_nd(phi, grass_indices)))))
            loss_02 = tf.reduce_mean(
                tf.square(tf.matmul(tf.gather_nd(phi, sky_indices), tf.transpose(tf.gather_nd(phi, horse_indices)))))
            loss_12 = tf.reduce_mean(
                tf.square(tf.matmul(tf.gather_nd(phi, grass_indices), tf.transpose(tf.gather_nd(phi, horse_indices)))))
            # total loss
            self.loss_convnet = loss_00 + loss_11 + loss_22 + loss_01 + loss_02 + loss_12
        elif loss_type == 2:
            class_gather_self = []
            class_gather_cross = []
            for i in range(nclass):
                class_idx = self.class_indices[i]
                gather_self = tf.gather_nd(self.phi[:, i], class_idx)
                cross_class_idx = [j for j in range(nclass) if j != i]
                gather_cross = tf.gather_nd(self.phi[:, cross_class_idx], class_idx)
                # _id = 0
                # for j in range(nclass):
                #     if i == j:
                #         continue
                #     gather_cross[:, _id] = gather[:, j]
                #     _id += 1
                class_gather_self.append(gather_self)
                class_gather_cross.append(gather_cross)
            self.loss_convnet = 0
            for i in range(nclass):
                loss = tf.reduce_mean(
                    tf.square(tf.matmul(class_gather_self[i], tf.transpose(class_gather_self[i])) - 1.0))
                self.loss_convnet += loss
                for j in range(i + 1, nclass):
                    loss = tf.reduce_mean(
                        tf.square(tf.matmul(class_gather_cross[i], tf.transpose(class_gather_cross[j]))))
                    self.loss_convnet += loss
        elif loss_type == 3:
            if nclass != 3:
                raise AssertionError('loss_type 3 can only be used with 3 classes')
            sky_indices = self.class_indices[0]
            grass_indices = self.class_indices[1]
            horse_indices = self.class_indices[2]

            sky_gather = tf.gather_nd(phi, sky_indices)
            horse_gather = tf.gather_nd(phi, grass_indices)
            grass_gather = tf.gather_nd(phi, horse_indices)

            sky_gather_self = tf.reshape(sky_gather[:, 0], (-1, 1))
            horse_gather_self = tf.reshape(horse_gather[:, 1], (-1, 1))
            grass_gather_self = tf.reshape(grass_gather[:, 2], (-1, 1))

            loss_00 = tf.reduce_mean(
                tf.square(
                    tf.matmul(sky_gather_self, tf.transpose(sky_gather_self)) - 1.0))
            loss_11 = tf.reduce_mean(
                tf.square(
                    tf.matmul(horse_gather_self, tf.transpose(horse_gather_self)) - 1.0))
            loss_22 = tf.reduce_mean(
                tf.square(
                    tf.matmul(grass_gather_self, tf.transpose(grass_gather_self)) - 1.0))

            sky_gather_cross_1 = tf.reshape(sky_gather[:, 1], (-1, 1))
            sky_gather_cross_2 = tf.reshape(sky_gather[:, 2], (-1, 1))

            horse_gather_cross_1 = tf.reshape(horse_gather[:, 0], (-1, 1))
            horse_gather_cross_2 = tf.reshape(horse_gather[:, 2], (-1, 1))

            grass_gather_cross_1 = tf.reshape(grass_gather[:, 0], (-1, 1))
            grass_gather_cross_2 = tf.reshape(grass_gather[:, 1], (-1, 1))

            # loss_01 = tf.reduce_mean(
            #     tf.square(tf.matmul(sky_gather_cross, tf.transpose(grass_gather_cross))))
            # loss_02 = tf.reduce_mean(
            #     tf.square(tf.matmul(sky_gather_cross, tf.transpose(horse_gather_cross))))
            # loss_12 = tf.reduce_mean(
            #     tf.square(tf.matmul(grass_gather_cross, tf.transpose(horse_gather_cross))))

            loss_01 = tf.reduce_mean(
                tf.square(tf.matmul(sky_gather_cross_1, tf.transpose(grass_gather_cross_1)) +
                          tf.matmul(sky_gather_cross_2, tf.transpose(grass_gather_cross_2))))
            loss_02 = tf.reduce_mean(
                tf.square(tf.matmul(sky_gather_cross_1, tf.transpose(horse_gather_cross_1)) +
                          tf.matmul(sky_gather_cross_2, tf.transpose(horse_gather_cross_1))))
            loss_12 = tf.reduce_mean(
                tf.square(tf.matmul(horse_gather_cross_1, tf.transpose(grass_gather_cross_1)) +
                          tf.matmul(horse_gather_cross_2, tf.transpose(grass_gather_cross_2))))

            # total loss
            self.loss_convnet = loss_00 + loss_11 + loss_22 + loss_01 + loss_02 + loss_12
        elif loss_type == 4:
            if nclass != 3:
                raise AssertionError('loss_type 4 can only be used with 3 classes')

            diff = self.Y - phi
            sky_indices = self.class_indices[0]
            grass_indices = self.class_indices[1]
            horse_indices = self.class_indices[2]

            sky_gather = tf.gather_nd(diff, sky_indices)
            horse_gather = tf.gather_nd(diff, grass_indices)
            grass_gather = tf.gather_nd(diff, horse_indices)

            self.loss_convnet = tf.nn.l2_loss(sky_gather) + tf.nn.l2_loss(horse_gather) + tf.nn.l2_loss(grass_gather)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.training_op = optimizer.minimize(self.loss_convnet)
