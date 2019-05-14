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

    public static Function NormalProbabilityFunction(Variable input, Variable mean, Variable std)
    {
        var constant2 = Constant.Scalar(DataType.Float, 2);
        var variance = CNTKLib.Pow(std, constant2);

        //probability
        var diff = CNTKLib.Minus(input, mean);
        var temp1 = CNTKLib.ElementTimes(diff, diff);
        temp1 = CNTKLib.ElementDivide(temp1, CNTKLib.ElementTimes(constant2, variance));
        temp1 = CNTKLib.Exp(CNTKLib.Negate(temp1));

        var temp2 = CNTKLib.ElementDivide(
            Constant.Scalar(DataType.Float, 1),
            CNTKLib.Sqrt(
                CNTKLib.ElementTimes(
                    variance, Constant.Scalar(DataType.Float, 2 * Mathf.PI))));
        return CNTKLib.ElementTimes(temp1, temp2);
    }

    public static float NormalProbability(float sample, float mean, float stddev)
    {
        var variance = stddev * stddev;
        var diff = sample - mean;
        var temp1 = diff * diff;
        temp1 = temp1 / (variance * 2.0f);
        temp1 = Mathf.Exp(-temp1);

        var temp2 = 1.0f / Mathf.Sqrt(variance * 2 * Mathf.PI);

        return temp1 * temp2;
    }

    //https://github.com/openai/spinningup/blob/master/spinup/exercises/problem_set_1_solutions/exercise1_2_soln.py
    public static Function NormalLogProbabilityLayer(Function mean, Function log_std, Variable value)
    {
        //pre_sum = -0.5 * (  ( (x-mu)/(tf.exp(log_std) + EPS) )**2 + 2*log_std + np.log(2*np.pi)  )
        var constant2 = Constant.Scalar(DataType.Float, 2.0f);
        
        var diff = CNTKLib.Minus(value, mean);
        var returnVal = CNTKLib.ElementDivide(diff, CNTKLib.Plus(CNTKLib.Exp(log_std), Constant.Scalar(DataType.Float, 0.0000001)));
        returnVal = CNTKLib.Pow(returnVal, constant2);

        returnVal = CNTKLib.Plus(returnVal, CNTKLib.ElementTimes(log_std, constant2));

        returnVal = CNTKLib.Plus(returnVal, Constant.Scalar(DataType.Float, Mathf.Log(2.0f * Mathf.PI)));

        returnVal = CNTKLib.Negate(CNTKLib.ElementDivide(returnVal, constant2));
        
        return returnVal;
    }

    public static float NormalLogProbability(float mean, float log_std, float value)
    {
        var diff = value - mean;

        var returnVal = diff / (Mathf.Exp(log_std) + 0.0000001f);
        returnVal = returnVal * returnVal;

        returnVal = returnVal + (2.0f * log_std);

        returnVal = returnVal + Mathf.Log(2.0f * Mathf.PI);

        return returnVal / -2.0f;
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

    public static float[] DiscountedRewards(float[] rewards, float discountFactor = 0.99f, float nextValue = 0)
    {
        float accum = nextValue;
        float[] result = new float[rewards.Length];
        for (int i = rewards.Length - 1; i >= 0; --i)
        {
            accum = accum * discountFactor + rewards[i];
            result[i] = accum;
        }

        return result;
    }

    public static float[] GeneralAdvantageEst(float[] rewards, float[] estimatedValues, float discountedFactor = 0.99f, float GAEFactor = 0.95f, float nextValue = 0)
    {
        Debug.Assert(rewards.Length == estimatedValues.Length);
        float[] deltaT = new float[rewards.Length];
        for (int i = 0; i < rewards.Length; ++i)
        {
            if (i != rewards.Length - 1)
            {
                deltaT[i] = rewards[i] + discountedFactor * estimatedValues[i + 1] - estimatedValues[i];
            }
            else
            {
                deltaT[i] = rewards[i] + discountedFactor * nextValue - estimatedValues[i];
            }

        }
        return DiscountedRewards(deltaT, GAEFactor * discountedFactor);
    }

    private static System.Random rng = new System.Random();
}
