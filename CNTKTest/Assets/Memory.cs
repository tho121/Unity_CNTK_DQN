using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

class Memory
{
    const float e = 0.01f;
    const float a = 0.6f;

    public Memory(int stateSize, int capacity = 1024)
    {
        m_experienceSize = (stateSize * 2) + 3;//current state, action, reward, next state, done

        m_memoryBuffer = new Queue<float>(capacity * m_experienceSize);
        m_randomIndexList = new int[capacity];

        for(int i = 0; i < capacity; ++i)
        {
            m_randomIndexList[i] = i;
        }

        m_tree = new SumTree(capacity);
    }

    public float GetPriority(float error)
    {
        return (float)Math.Pow((error + e), a);
    }

    public void Add(float error, float[] sample)
    {
        float p = GetPriority(error);
        m_tree.Add(p, sample);
    }

    public float[] GetSamples(int sampleSize, out int[] indexes)
    {
        indexes = new int[sampleSize];
        List<float> experiences = new List<float>(m_experienceSize * sampleSize);

        float segment = m_tree.Total() / sampleSize;

        for (int i = 0; i < sampleSize; ++i)
        {
            var a = segment * i;
            var b = segment * (i + 1);
            var s = UnityEngine.Random.Range(a, b);

            float outTreeVal;
            var experience = m_tree.Get(s, out indexes[i], out outTreeVal);
            experiences.AddRange(experience);
        }

        return experiences.ToArray();
    }

    public void Update(int idx, float error)
    {
        var p = GetPriority(error);
        m_tree.Update(idx, p);
    }

    //public void Add(float[] experience)
    //{
    //    //array should be size of m_inputSize
    //    Debug.Assert(experience.Length == m_experienceSize);

    //    for(int i = 0; i < experience.Length; ++i)
    //    {
    //        m_memoryBuffer.Enqueue(experience[i]);
    //    }
    //}

    //public float[] GetSamples(int sampleSize)
    //{
    //    float[] allSamples = m_memoryBuffer.ToArray();

    //    //if requested sample size is greater than the current memory size
    //    var currentMemoryCount = m_memoryBuffer.Count / m_experienceSize;
    //    if (sampleSize > currentMemoryCount)
    //    {
    //        return allSamples;
    //    }

    //    ShuffleClass.Shuffle<int>(m_randomIndexList);

    //    List<float> samples = new List<float>(sampleSize * m_experienceSize);

    //    int randomIndex = 0;
    //    for(int i = 0; i < sampleSize; ++i)
    //    {
    //        while(randomIndex < m_randomIndexList.Length && m_randomIndexList[randomIndex] >= currentMemoryCount)
    //        {
    //            randomIndex++;
    //        }

    //        int index = m_randomIndexList[randomIndex] * m_experienceSize;

    //        for (int j = 0; j < m_experienceSize; ++j)
    //        {
    //            samples.Add(allSamples[index + j]);
    //        }

    //        randomIndex++;
    //    }

    //    return samples.ToArray();
    //}

    private Queue<float> m_memoryBuffer;
    private int m_experienceSize;
    private int[] m_randomIndexList;

    private SumTree m_tree;
}

static class ShuffleClass
{
    private static Random rng = new Random();

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
}
