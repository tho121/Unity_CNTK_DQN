using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using CNTK;

public static class Utils {

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
    public static Function GaussianLogProbabilityLayer(Function mean, Function std, Variable value)
    {
        var constant2 = Constant.Scalar(DataType.Float, 2);
        var pdfDom = CNTKLib.ElementTimes(CNTKLib.Pow(std, constant2), Constant.Scalar(DataType.Double, System.Math.PI * 2));
        pdfDom = CNTKLib.Pow(pdfDom, Constant.Scalar(DataType.Float, 0.5));

        var pdfNum = CNTKLib.Pow(CNTKLib.Minus(value, mean), constant2);

        //no division, pow to -1 to create recipocal then multiply
        var temp0 = CNTKLib.Pow(std, constant2);
        var temp = CNTKLib.Pow(temp0, Constant.Scalar(DataType.Float, -1));
        pdfNum = CNTKLib.ElementTimes(pdfNum, temp);

        pdfNum = CNTKLib.ElementTimes(pdfNum, Constant.Scalar(DataType.Float, -0.5f));
        pdfNum = CNTKLib.Exp(pdfNum);

        var pdf = CNTKLib.ElementTimes(pdfNum, CNTKLib.Pow(pdfDom, Constant.Scalar(DataType.Float, -1)));

        return pdf;
    }

    private static System.Random rng = new System.Random();
}
