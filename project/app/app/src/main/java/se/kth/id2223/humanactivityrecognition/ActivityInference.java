package se.kth.id2223.humanactivityrecognition;

import android.content.Context;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * Base class used for predicting using TensorFlow java library
 */
public class ActivityInference {

  static {
    System.loadLibrary("tensorflow_inference");
  }

  private TensorFlowInferenceInterface mInferenceInterface;

  private final static String PATH_TO_MODEL = "file:///android_asset/frozen_har.pb";
  // Should be defined in training script
  // Input node name, i.e. placeholder's name for input of the data
  private final static String INPUT_NODE_NAME = "input";
  // Output node name, the one that computes softmax activation on logits
  private final static String OUTPUT_NODE_NAME = "y_";
  // Input size, num_examples x num_features
  private final static long[] INPUT_SIZE = {1, 200, 3};
  private static final String[] OUTPUT_NODES = {"y_"};
  // Output size, number of target labels
  private final static int NUM_OUTPUT_CLASSSES = 6;

  public ActivityInference(final Context context){
    mInferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), PATH_TO_MODEL);
  }

  public float[] getActivityProbabilities(float [] inputSignal){
    float[] result = new float[NUM_OUTPUT_CLASSSES];
    mInferenceInterface.feed(INPUT_NODE_NAME,inputSignal,INPUT_SIZE);
    mInferenceInterface.run(OUTPUT_NODES);
    mInferenceInterface.fetch(OUTPUT_NODE_NAME,result);
    return result;
  }
}
