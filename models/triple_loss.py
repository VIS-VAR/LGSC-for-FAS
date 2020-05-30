import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L


class TripletLoss(fluid.dygraph.Layer):
    """Triplet Loss

    arXiv: https://arxiv.org/pdf/1703.07737.pdf
    """

    def __init__(self, type='BatchAll', margin=0.5, loss_weight=1.0):
        super(TripletLoss, self).__init__()
        if type not in ['BatchAll', 'BatchHard']:
            raise TypeError
        self.type = type
        self.magin = margin
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        target = 1 - target[:, 0]
        batch_size, vector_size = pred.shape[0], pred.shape[1]

        pred = L.l2_normalize(pred, axis=1, epsilon=1e-10)

        square_norm = L.reduce_sum(L.square(pred), dim=1)
        dist = L.elementwise_add(-2.0 * L.matmul(pred, pred, transpose_y=True), square_norm, axis=0)
        dist = L.elementwise_add(dist, square_norm, axis=1)
        dist = L.elementwise_max(dist, L.zeros_like(dist))
        dist = L.sqrt(dist)

        ap_dist = L.reshape(dist, (0, 0, 1))
        an_dist = L.reshape(dist, (0, 1, -1))

        loss = L.expand(ap_dist, (1, 1, batch_size)) - L.expand(an_dist, (1, batch_size, 1)) + self.magin

        indice_equal = L.diag(L.fill_constant((batch_size,), dtype='float32', value=1.0))
        indice_not_equal = 1.0 - indice_equal

        broad_matrix = L.expand(L.reshape(target, (-1, 1)), (1, batch_size)) + L.expand(
            L.reshape(target, (1, -1)), (batch_size, 1))

        pp = L.cast(L.equal(broad_matrix, L.zeros_like(broad_matrix)), dtype='float32')
        pp = L.reshape(indice_not_equal * pp, (0, 0, 1))
        
        pn = L.cast(L.equal(broad_matrix, L.zeros_like(broad_matrix) + 1), dtype='float32')
        pn = L.reshape(indice_not_equal * pn, (1, 0, -1))
        
        apn = L.expand(pp, (1, 1, batch_size)) * L.expand(pn, (batch_size, 1, 1))
        
        loss = loss * L.cast(apn, dtype='float32')
        loss = L.elementwise_max(loss, L.zeros_like(loss))

        num_tri = L.reduce_sum(L.cast(L.greater_than(loss, L.zeros_like(loss)), dtype='float32'))

        loss = L.reduce_sum(loss) * self.loss_weight / (num_tri + 1e-16)

        return loss


class PairLoss(fluid.dygraph.Layer):

    def __init__(self, type='BatchAll', margin=0.5, loss_weight=1.0):
        super(PairLoss, self).__init__()
        if type not in ['BatchAll', 'BatchHard']:
            raise TypeError
        self.type = type
        self.magin = margin
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        target = 1 - target[:, 0]
        batch_size, vector_size = pred.shape[0], pred.shape[1]

        pred = L.l2_normalize(pred, axis=1, epsilon=1e-10)

        square_norm = L.reduce_sum(L.square(pred), dim=1)
        dist = L.elementwise_add(-2.0 * L.matmul(pred, pred, transpose_y=True), square_norm, axis=0)
        dist = L.elementwise_add(dist, square_norm, axis=1)
        dist = L.elementwise_max(dist, L.zeros_like(dist))
        dist = L.sqrt(dist)

        indice_equal = L.diag(L.fill_constant((batch_size,), dtype='float32', value=1.0))
        indice_not_equal = 1.0 - indice_equal

        broad_matrix = L.expand(L.reshape(target, (-1, 1)), (1, batch_size)) + L.expand(
            L.reshape(target, (1, -1)), (batch_size, 1))
        
        pn = L.cast(L.equal(broad_matrix, L.zeros_like(broad_matrix) + 1), dtype='float32')
        pn = indice_not_equal * pn
        
        loss = (self.magin - dist) * L.cast(pn, dtype='float32')
        loss = L.elementwise_max(loss, L.zeros_like(loss))

        num_tri = L.reduce_sum(L.cast(L.greater_than(loss, L.zeros_like(loss)), dtype='float32'))

        loss = L.reduce_sum(loss) * self.loss_weight / (num_tri + 1e-16)

        return loss
