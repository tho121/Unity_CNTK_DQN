using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;

public class VPGActor {

    const float Epsilon = 0.01f;

	public VPGActor(int stateSize, int actionSize, int layerSize)
    {
        for (int i = 0; i < actionSize; ++i)
        {
            m_normDist.Add(new EDA.Gaussian());
        }

        //network
        CreateNetwork(stateSize, actionSize, layerSize);

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

        m_error = CNTKLib.InputVariable(new int[] { 2 }, DataType.Float);

        var layer1 = Layer(m_inputVariable, layerSize);
        var layer2 = Layer(layer1, layerSize);
        var layer3 = Layer(layer2, layerSize);

        var model = new VariableVector();
        var losslist = new VariableVector();

        for (int i = 0; i < actionSize; ++i)
        {
            var mean = Layer(layer3, 1);
            var std = Layer(layer3, 1);
            m_mean.Add(mean);
            m_std.Add(std);

            model.Add(mean);
            model.Add(std);

            m_normalSamples.Add(CNTKLib.InputVariable(new int[] { 1 }, DataType.Float));

            var logProb = Utils.GaussianLogProbabilityLayer(m_mean[i], m_std[i], m_normalSamples[i]);

            var lossFunc = CNTKLib.Plus(CNTKLib.ElementTimes(CNTKLib.Negate(logProb), m_error), Constant.Scalar(DataType.Float, Epsilon));
            losslist.Add(lossFunc);
        }

        m_loss = CNTKLib.Sum(losslist);
        m_model = CNTKLib.Splice(model, new Axis(0));
    }

    public float[] Act(float[] state, DeviceDescriptor device)
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


        //
        m_normDist[0].Mean = denseDataFloat[0];
        m_normDist[0].StdDev = denseDataFloat[1];
        m_normDist[1].Mean = denseDataFloat[2];
        m_normDist[1].StdDev = denseDataFloat[3];

        return new float[2] { (float)m_normDist[0].Next(), (float)m_normDist[1].Next() };
    }

    public void Train(float[] state, float[] errors, float[] samples, DeviceDescriptor device)
    {
        //m_error = error[0]
        //m_sampleNormalVariable = samples[0]

        var loss = errors;

        Value stateData = Value.CreateBatch<float>(new int[] { state.Length }, state, device);
        Value sample1 = Value.CreateBatch<float>(new int[] { 1 }, new float[1] { samples[0] }, device);
        Value sample2 = Value.CreateBatch<float>(new int[] { 1 }, new float[1] { samples[1] }, device);
        Value output = Value.CreateBatch<float>(new int[] { errors.Length }, errors, device);

        var arguments = new Dictionary<Variable, Value>()
            {
                { m_inputVariable, stateData },
                { m_normalSamples[0],  sample1},
                { m_normalSamples[1],  sample2},
                { m_error, output}
            };
        try
        {
            m_trainer.TrainMinibatch(arguments, false, device);
        }
        catch(System.Exception e)
        {
            Debug.Log(e.Message);
        }
    }

    private Variable m_inputVariable;
    private List<Variable> m_normalSamples = new List<Variable>();
    private Variable m_error;
    private Function m_model;
    private Function m_loss;

    private List<Function> m_lossList = new List<Function>();

    private List<Function> m_mean = new List<Function>();
    private List<Function> m_std = new List<Function>();

    private List<EDA.Gaussian> m_normDist = new List<EDA.Gaussian>();

    private Trainer m_trainer;
}
