using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;

public class VPGCritic {

    public VPGCritic(int stateSize, int actionSize, int layerSize)
    {
        //network
        CreateNetwork(stateSize, actionSize, layerSize);

        m_targetVariable = CNTKLib.InputVariable(new int[] { actionSize }, DataType.Float);

        //loss
        m_loss = CNTKLib.Square(CNTKLib.Minus(m_model.Output, m_targetVariable));

        var learningRate = new TrainingParameterScheduleDouble(0.01);
        var learner = new List<Learner>() { Learner.SGDLearner(m_model.Parameters(), learningRate) };

        m_trainer = Trainer.CreateTrainer(m_model, m_loss, null, learner);

    }

    private Function Layer(Variable input, int outputCount)
    {
        Parameter bias = new Parameter(new int[] { outputCount }, DataType.Float, 0);
        Parameter weights = new Parameter(new int[] { outputCount, input.Shape[0] }, DataType.Float,
            CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale));

        var z = CNTKLib.Times(weights, input);
        z = CNTKLib.Plus(bias, z);

        Function activation = CNTKLib.Softplus(z, "Softplus");
        return activation;
    }

    private void CreateNetwork(int stateSize, int actionSize, int layerSize)
    {
        m_inputVariable = CNTKLib.InputVariable(new int[] { stateSize }, DataType.Float);

        var layer1 = Layer(m_inputVariable, layerSize);
        var layer2 = Layer(layer1, layerSize);
        var layer3 = Layer(layer2, actionSize); //each action space will have a mean and std calculated

        m_model = layer3;
    }

    public float[] Predict(float[] state, DeviceDescriptor device)
    {
        Value input = Value.CreateBatch<float>(new int[] { state.Length }, state, device);

        var inputDict = new Dictionary<Variable, Value>()
        {
            { m_inputVariable, input },
        };

        var outputDict = new Dictionary<Variable, Value>()
        {
            { m_model.Output, null },
        };

        m_model.Evaluate(inputDict, outputDict, device);

        var outputValue = outputDict[m_model.Output];
        var denseData = outputValue.GetDenseData<float>(m_model.Output)[0];

        float[] denseDataFloat = new float[denseData.Count];
        denseData.CopyTo(denseDataFloat, 0);

        return denseDataFloat;
    }

    public void Train(float[] state, float[] targets, DeviceDescriptor device)
    {
        Value input = Value.CreateBatch<float>(new int[] { state.Length }, state, device);
        Value output = Value.CreateBatch<float>(new int[] { targets.Length }, targets, device);

        var arguments = new Dictionary<Variable, Value>()
            {
                { m_inputVariable, input },
                { m_targetVariable, output}
            };

        m_trainer.TrainMinibatch(arguments, false, device);
    }

    private Variable m_inputVariable;
    private Variable m_targetVariable;
    private Function m_model;
    private Function m_loss;
    private Trainer m_trainer;
}
