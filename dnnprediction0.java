/**
 * Created by tengben on 2017/11/6.
 */

import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.SavedModelBundle;
import java.nio.FloatBuffer;

public class dnnprediction0 {
        public static void main(String[] args) throws Exception {
            SavedModelBundle bundle=SavedModelBundle.load("D:/DNNClassfierCreditcard","serve");
            Session s = bundle.session();

            double[] inputDouble = {0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62};
            float [] inputfloat=new float[inputDouble.length];
            for(int i=0;i<inputfloat.length;i++)
            {
                inputfloat[i]=(float)inputDouble[i];
            }
            //Tensor inputTensor = Tensor.create(new long[] {35}, FloatBuffer.wrap(inputfloat) );

            FloatBuffer.wrap(inputfloat) ;
            float[][] data= new float[1][30];
            data[0]=inputfloat;
            Tensor inputTensor=Tensor.create(data);

            Tensor result = s.runner()
                    .feed("dnn/input_from_feature_columns/input_from_feature_columns/concat", inputTensor)
                    //.feed("input_example_tensor", inputTensor)
                    //.fetch("tensorflow/serving/classify")
                    .fetch("dnn/binary_logistic_head/predictions/probabilities")
                    //.fetch("dnn/zero_fraction_3/Cast")
                    .run().get(0);


            float[][] m = new float[1][2];
            float[][] vector = (float[][])result.copyTo(m);
            float maxVal = 0;
            int inc = 0;
            int predict = -1;
            for(float val : vector[0])
            {
                   // System.out.println(val+"  ");
                    if(val > maxVal) {
                        predict = inc;
                        maxVal = val;
                    }
                    inc++;
            }
            System.out.println(predict);

        }
}

