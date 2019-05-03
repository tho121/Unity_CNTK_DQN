using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using CNTK;

public static class Utils {

    public static Function Layer(Variable input, int outputCount)
    {
        Parameter bias = new Parameter(new int[] { outputCount }, DataType.Float, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale));
        Parameter weights = new Parameter(new int[] { outputCount, input.Shape[0] }, DataType.Float,
            CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale));

        var z = CNTKLib.Times(weights, input);
        z = CNTKLib.Plus(bias, z);
        return z;
    }

    //credit
    //https://gist.github.com/tansey/1444070
    public static double SampleGaussian(double mean, double stddev)
    {
        // The method requires sampling from a uniform random of (0,1]
        // but Random.NextDouble() returns a sample of [0,1).
        double x1 = 1 - rng.NextDouble();
        double x2 = 1 - rng.NextDouble();

        double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
        return y1 * stddev + mean;
    }

    //credit
    //https://stackoverflow.com/questions/273313/randomize-a-listt
    public static void Shuffle<T>(this IList<T> list)
    {
        int n = list.Count;
        while (n > 1)
        {
            n--;
            int k = rng.Next(n + 1);
            T value = list[k];
            list[k] = list[n];
            list[n] = value;
        }
    }

    //https://www.tensorflow.org/api_docs/python/tf/distributions/Normal
    public static Function GaussianProbabilityLayer(Function mean, Function std, Variable value)
    {
        var constant2 = Constant.Scalar(DataType.Float, 2);
        var diff = CNTKLib.Minus(value, mean);
        var temp1 = CNTKLib.ElementTimes(diff, diff);
        temp1 = CNTKLib.ElementDivide(temp1, CNTKLib.ElementTimes(constant2, std)); // CNTKLib.ElementTimes(std, std)
        temp1 = CNTKLib.Exp(CNTKLib.Negate(temp1));

        var temp2 = CNTKLib.ElementDivide(
            Constant.Scalar(DataType.Float, 1),
            CNTKLib.Sqrt(
                CNTKLib.ElementTimes(
                    std, Constant.Scalar(DataType.Float, 2 * Mathf.PI)))); // CNTKLib.ElementTimes(std, std)
        return CNTKLib.ElementTimes(temp1, temp2);
    }

    public static float[] CalculateGAE(float[] rewards, float[] targetValues, float discountFactor, float finalValue = 0.0f)
    {
        float[] advantages = new float[rewards.Length];

        for(int i = 0; i < rewards.Length; ++i)
        {
            if(i + 1 < rewards.Length)
            {
                advantages[i] = rewards[i] - targetValues[i] + (discountFactor * targetValues[i + 1]);
            }
            else
            {
                advantages[i] = rewards[i] - targetValues[i] + (discountFactor * finalValue);
            }
        }

        float[] rewardsTotal = new float[rewards.Length];
        float rewardTotal = 0.0f;

        for (int i = rewards.Length - 1; i >=0; --i)
        {
            rewardsTotal[i] = rewardTotal = rewardTotal * discountFactor + advantages[i];
        }

        return rewardsTotal;
    }

    private static System.Random rng = new System.Random();
}
