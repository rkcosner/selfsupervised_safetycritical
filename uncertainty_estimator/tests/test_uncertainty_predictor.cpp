//
// Created by ivan on 9/2/21.
//

#include <iostream>
#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>
#include <memory>
#include "uncertainty_predictor.h"


std::vector<at::Tensor> clone_paramter_vector(const torch::jit::script::Module & module){
  std::vector<at::Tensor> paramlist;
  for (const auto &parameter: module.parameters()) {
    paramlist.push_back(parameter.clone());
  }
  return paramlist;
}

using ImageMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ErrorTensor = Eigen::Tensor<float, 3, Eigen::RowMajor>;

UncertaintyPredictor<200,320, 65> unc_pred_ptr;
ImageMat left;
ImageMat right;
ImageMat  error;
std::vector<torch::Tensor> pretrain_params = clone_paramter_vector(unc_pred_ptr.model);

//std::unique_ptr<ImageMat> left(new ImageMat());
//std::unique_ptr<ImageMat> right(new ImageMat());
//ImageMat left = ImageMat::Ones();
//ImageMat right = ImageMat::Ones();

TEST(test_uncertainty_predictor, test_call) {
  left = ImageMat::Zero(200, 320);
  right = ImageMat::Zero(200, 320);

  auto eigTensor = unc_pred_ptr.make_uncertainty_prediction(left, right);
//  std::cout << "unc_pred_ptr.last_tensor.sizes():"  << std::endl << unc_pred_ptr.last_tensor.sizes() << std::endl;
//  for(const auto& attribute: unc_pred_ptr.model.named_attributes()){
//    std::cout << "attribute.name:"  << std::endl << attribute.name << std::endl;
//  }
  std::cout << "I Build."  << std::endl;
}

TEST(test_uncertainty_predictor, test_train_call) {
  left = ImageMat::Zero(200, 320);
  right = ImageMat::Zero(200, 320);
  error = ImageMat::Random(200, 320);
  error.setRandom();
  auto eigTensor = unc_pred_ptr.make_uncertainty_prediction_and_refine(left, right, error);
  auto current_parameters = make_paramter_vector(unc_pred_ptr.model);
  for (int param_idx = 0; param_idx < current_parameters.size(); ++param_idx) {
    std::cout << "idx_max_norm: "  << std::endl << (current_parameters[param_idx] - pretrain_params[param_idx]).flatten().norm(2).max().item().toFloat() << std::endl;
  }
//  std::cout << "unc_pred_ptr.last_tensor.sizes():"  << std::endl << unc_pred_ptr.last_tensor.sizes() << std::endl;
//  for(const auto& attribute: unc_pred_ptr.model.named_attributes()){
//    std::cout << "attribute.name:"  << std::endl << attribute.name << std::endl;
//  }
  std::cout << "I Build."  << std::endl;
}
