package se.kth.spark.lab1.task6

import org.apache.spark.ml.{PredictionModel, Predictor}

abstract class MyLinearRegression[
    FeaturesType,
    Learner <: MyLinearRegression[FeaturesType, Learner, Model],
    Model <: MyLinearModel[FeaturesType, Model]]
  extends Predictor[FeaturesType, Learner, Model] {
}

abstract class MyLinearModel[FeaturesType, Model <: MyLinearModel[FeaturesType, Model]]
  extends PredictionModel[FeaturesType, Model] {
}