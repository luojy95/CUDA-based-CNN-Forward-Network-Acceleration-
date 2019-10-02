
#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */

    /*
    Data layout:
    y: output data, batch size * output channels * y * x
    x: input data, batch size * input channels * y * x
    k: kernel weights, output channels * input channels * y * x
    */

    const int B = x.shape_[0]; // batch size
    const int M = y.shape_[1]; // output channels
    const int C = x.shape_[1]; // input channels
    const int H = x.shape_[2]; // image height
    const int W = x.shape_[3]; // image width
    const int K = k.shape_[3]; // kernel size

    for (int b = 0; b < B; ++b) { // for each image in the batch

        // CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";

        /* ... a bunch of nested loops later...
            y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
        */

        for (int m = 0; m < M; m++) { // for each output feature maps
            for (int h = 0; h < H - K + 1; h++) { // for each output element
                for (int w = 0; w < W - K + 1; w++) {
                    float result = 0;
                    for (int c = 0; c < C; c++) { // sum over all input feature maps
                        for (int p = 0; p < K; p++) { // for each kernel weight
                            for (int q = 0; q < K; q++) {
                                result += x[b][c][h+p][w+q] * k[m][c][p][q];
                                // Unlike the convolutions described in the class, note that this one is NOT centered on the input image. It's a forward kernel rather than center kernel. Note H-K+1 and W-K+1!
                            }
                        }
                    }
                    y[b][m][h][w] = result;
                }
            }
        }
    }

}
}
}

#endif