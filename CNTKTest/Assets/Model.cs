using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;


class Model
{
    private static Function LinearLayer(Variable input, int outputCount)
    {
        Parameter bias = new Parameter(new int[] { outputCount }, DataType.Float, 0);
        Parameter weights = new Parameter(new int[] { outputCount, input.Shape[0]}, DataType.Float,
            CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale, CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 0));

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

        return linear3;
    }
}

