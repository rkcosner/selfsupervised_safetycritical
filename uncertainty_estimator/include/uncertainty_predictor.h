//
// Created by ivan on 9/2/21.
//

#ifndef FULL_STACK_UNCERTAINTY_PREDICTOR_H
#define FULL_STACK_UNCERTAINTY_PREDICTOR_H

#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "torch_utils.h"

using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using TensorXfRow = Eigen::Tensor<float, 3, Eigen::RowMajor>;

std::vector<at::Tensor> make_paramter_vector(const torch::jit::script::Module & module){
  std::vector<at::Tensor> paramlist;
  for (const auto &parameter: module.parameters()) {
    paramlist.push_back(parameter);
  }
  return paramlist;
}

template<int height, int width, int num_classes>
class UncertaintyPredictor {
 public:
  UncertaintyPredictor(float lr=1e-3, int batch_size=1) :
  model(torch::jit::load("/home/drew/Aeronvironment/data/prod_mod.th")),
  device(torch::kCUDA),
  last_tensor(torch::zeros({height,width, num_classes})),
  loss_func(),
  optimizer(make_paramter_vector(model), torch::optim::AdamOptions(lr)),
  batch_size(batch_size),
  batch_buffer(torch::zeros({batch_size, 2, height, width}).to(device)),
  batch_error_tensor_buffer(torch::zeros({batch_size, height, width}).to(device)),
  batch_insert_idx(0),
  batch_full(false)
//  ,
//  loss_model(torch::nn::)
  {
    model.to(device);
  }


  Eigen::Tensor<float, 3, Eigen::RowMajor> make_uncertainty_prediction(
    const MatrixXfRow& left,
    const MatrixXfRow& right) {
    torch::NoGradGuard no_grad;
    model.eval();
    //make row major
    std::cout << "[INFO] Normalizing" << std::endl;
    torch::Tensor left_tensor = (dynEigenToTorch<height, width>(left).unsqueeze(0)-0.5)*2.0 ;
    torch::Tensor right_tensor = (dynEigenToTorch<height, width>(right).unsqueeze(0)-0.5)*2.0;
    
    // std::cout << "[INFO] No normalization." << std::endl;
    // torch::Tensor left_tensor = dynEigenToTorch<height, width>(left).unsqueeze(0);
    // torch::Tensor right_tensor = dynEigenToTorch<height, width>(right).unsqueeze(0);
    auto images = torch::stack(std::vector<torch::Tensor>{left_tensor, right_tensor}, 1).to(device);
    torch::Dict<std::string, torch::Tensor> forward_input;
    forward_input.template insert("images", images);
    auto model_output = model.forward(std::vector<torch::jit::IValue>{forward_input}).toTensor().squeeze();
    last_tensor = torch::softmax(torch::permute(model_output, {1,2,0}).contiguous(),
                                 2).to(torch::kCPU);
    return torchToDynEigenTensor<height, width,  num_classes>(last_tensor);
  }

  Eigen::Tensor<float, 3, Eigen::RowMajor> make_uncertainty_prediction_and_refine(
    const MatrixXfRow& left,
    const MatrixXfRow& right,
    const MatrixXfRow& error_measurement) {
    model.train();
    //Prepare training
    optimizer.zero_grad();

    //make row major
    std::cout << "[INFO] Normalizing" << std::endl;
    torch::Tensor left_tensor = (dynEigenToTorch<height, width>(left).unsqueeze(0)-0.5)*2.0;
    torch::Tensor right_tensor = (dynEigenToTorch<height, width>(right).unsqueeze(0)-0.5)*2.0;

    // std::cout << "[INFO] No normalization." << std::endl;
    // torch::Tensor left_tensor = dynEigenToTorch<height, width>(left).unsqueeze(0);
    // torch::Tensor right_tensor = dynEigenToTorch<height, width>(right).unsqueeze(0);
    torch::Tensor error_tensor = dynEigenToTorch<height, width>(error_measurement);

    //forward pass
    auto images = torch::stack(std::vector<torch::Tensor>{left_tensor, right_tensor}, 1).to(device);
    torch::Dict<std::string, torch::Tensor> forward_input;
    forward_input.template insert("images", images);
    auto model_output = model.forward(std::vector<torch::jit::IValue>{forward_input}).toTensor().squeeze();

    //training
    auto labels = error_tensor.toType(c10::ScalarType::Long).to(device);
    auto true_loss = loss_func(model_output.unsqueeze(0),
                                       labels.unsqueeze(0));
    true_loss.backward();
    optimizer.step();

     last_tensor = torch::softmax(torch::permute(model_output, {1,2,0}).contiguous(),
                                   2).to(torch::kCPU);
    return torchToDynEigenTensor<height, width,  num_classes>(last_tensor);
  }

  Eigen::Tensor<float, 3, Eigen::RowMajor> batch_make_uncertainty_prediction_and_refine(
    const MatrixXfRow& left,
    const MatrixXfRow& right,
    const MatrixXfRow& error_measurement) {
    using namespace torch::indexing;
    model.train();
    //Prepare training
    optimizer.zero_grad();

    //make row major
    std::cout << "[INFO] Normalizing" << std::endl;
    torch::Tensor left_tensor = (dynEigenToTorch<height, width>(left).unsqueeze(0)-0.5)*2.0;
    torch::Tensor right_tensor = (dynEigenToTorch<height, width>(right).unsqueeze(0)-0.5)*2.0;

    // std::cout << "[INFO] No normalization." << std::endl;
    // torch::Tensor left_tensor = dynEigenToTorch<height, width>(left).unsqueeze(0);
    // torch::Tensor right_tensor = dynEigenToTorch<height, width>(right).unsqueeze(0);
    torch::Tensor error_tensor = dynEigenToTorch<height, width>(error_measurement);

    //forward pass
    auto images = torch::stack(std::vector<torch::Tensor>{left_tensor, right_tensor}, 1).to(device);
    batch_buffer.index({batch_insert_idx}) = images.index({0});
    batch_error_tensor_buffer.index({batch_insert_idx}) = error_tensor.index({0}).to(device);
    int insert_idx = batch_insert_idx;
    // std::cout << "[INFO] batch insert idx: " << batch_insert_idx << std::endl;
    batch_insert_idx++;
    batch_insert_idx %= batch_size;
    if(batch_insert_idx == 0) {
      batch_full = true;
    }
    if(!batch_full) {
      // std::cout << "[INFO]  Unfilled Batch" << std::endl;
      return make_uncertainty_prediction(left, right);
    }

    torch::Dict<std::string, torch::Tensor> forward_input;
    // std::cout << "[INFO]  Images Shape: " << images.sizes() << std::endl;
    // std::cout << "[INFO]  Buffer Shape: " << batch_buffer.sizes() << std::endl;
    // std::cout << "[INFO]  Buffer Error Shape: " << batch_error_tensor_buffer.sizes() << std::endl;
    forward_input.template insert("images", batch_buffer);
    auto model_output = model.forward(std::vector<torch::jit::IValue>{forward_input}).toTensor();

    //training
    auto labels = batch_error_tensor_buffer.toType(c10::ScalarType::Long);
    auto true_loss = loss_func(model_output,
                                       labels);
    true_loss.backward();
    optimizer.step();

    last_tensor = torch::softmax(torch::permute(model_output.index({insert_idx}), {1,2,0}).contiguous(), 2).to(torch::kCPU);
    return torchToDynEigenTensor<height, width,  num_classes>(last_tensor);
  }

  torch::jit::script::Module model;
  torch::DeviceType device;
  torch::Tensor last_tensor;
  torch::nn::CrossEntropyLoss loss_func;
  torch::optim::Adam optimizer;
  int batch_size;
  torch::Tensor batch_buffer;
  torch::Tensor batch_error_tensor_buffer;
  int batch_insert_idx;
  bool batch_full;

};

#endif //FULL_STACK_UNCERTAINTY_PREDICTOR_H
