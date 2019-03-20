using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;


class Model
{
    private static Function LinearLayer(Variable input, int outputCount)
    {
        Parameter bias = new Parameter(new int[] { outputCount }, DataType.Float, 0);
        Parameter weights = new Parameter(new int[] { outputCount, input.Shape[0] }, DataType.Float,
            CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale));

        var z = CNTKLib.Times(weights, input);
        z = CNTKLib.Plus(bias, z);
        Function sigmoid = CNTKLib.Sigmoid(z, "LinearSigmoid");
        return sigmoid;
    }

    public static Function CreateNetwork(int stateSize, int actionSize, int layerSize, out Variable inputVariable)
    {
        inputVariable = CNTKLib.InputVariable(new int[] { stateSize }, DataType.Float);

        var linear1 = LinearLayer(inputVariable, layerSize);
        var relu1 = CNTKLib.ReLU(linear1);
        var linear2 = LinearLayer(relu1, layerSize);
        var relu2 = CNTKLib.ReLU(linear2);
        var linear3 = LinearLayer(relu2, actionSize);
        var relu3 = CNTKLib.ReLU(linear3);
        var linear4 = LinearLayer(relu3, actionSize);

        return linear4;
    }

    public static void SoftUpdate(Function trainingNetwork, Function targetNetwork, DeviceDescriptor device, float updateRate = 0.001f)
    {
        var trainingParams = trainingNetwork.Parameters();
        var targetParams = targetNetwork.Parameters();

        Debug.Assert(trainingParams.Count == targetParams.Count);

        for (int i = 0; i < targetParams.Count; ++i)
        {
            var targetValues = new Value(targetParams[i].GetValue());
            var targetData = targetValues.GetDenseData<float>(targetParams[i]);

            var trainingValues = new Value(trainingParams[i].GetValue());
            var trainingData = trainingValues.GetDenseData<float>(trainingParams[i]);

            List<float> values = new List<float>(targetValues.Shape.TotalSize);

            for(int j = 0; j < targetData.Count; ++j)
            {
                for(int k = 0; k < targetData[j].Count; ++k)
                {
                    var val = targetData[j][k];

                    values.Add((val * (1.0f - updateRate)) + (trainingData[j][k] * updateRate));
                }
            }

            var data = new NDArrayView(targetValues.Shape, values.ToArray(), device);
            
            targetParams[i].SetValue(data);
        }
    }
}

