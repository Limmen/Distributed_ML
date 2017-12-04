package se.kth.id2223.humanactivityrecognition;

import android.content.Context;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class used for predicting using TensorFlow java library
 */
public class ActivityInference {

  static {
    System.loadLibrary("tensorflow_inference");
  }

  private TensorFlowInferenceInterface mInferenceInterface;
  private final String mPathToModel;
  private final String mInputNodeName;
  private final String [] mOutputNodes;
  private final String mOutputNodeName;
  private final long[] mInputSize;
  private final int mOutputSize;
  private final List<Integer> mPredictions;
  private final static String sPathToModel = "file:///android_asset/model_name.pb";

  // Should be defined in training script
  // Input node name, i.e. placeholder's name for input of the data
  private final static String sInputNodeName = "input";
  // Output node name, the one that computes softmax activation on logits
  private final static String sOutPutNodeName = "output";
  // Input size, num_examples x num_features
  private final static long[] sInputSize = {1, 32};
  // Output size, number of target labels
  private final static int sOutputSize = 2;

  public ActivityInference(String inputNodeName, String outputNodeName, long[] inputSize, int outputSize, String pathToModel, final Context context){
    mPredictions = new ArrayList<>();
    mPathToModel = pathToModel;
    mInputNodeName = inputNodeName;
    mOutputNodes = new String[]{outputNodeName};
    mOutputNodeName = outputNodeName;
    mInputSize = inputSize;
    mOutputSize = outputSize;
    mInferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), mPathToModel);
  }

  public float[] getActivityProbabilities(float [] inputSignal){
    float[] result = new float[mOutputSize];
    mInferenceInterface.feed(mInputNodeName,inputSignal,mInputSize);
    mInferenceInterface.run(mOutputNodes);
    mInferenceInterface.fetch(mOutputNodeName,result);
    return result;
  }
}
