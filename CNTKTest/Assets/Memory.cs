using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

class Memory
{
    public Memory(int stateSize, int capacity = 1024)
    {
        m_experienceSize = (stateSize * 2) + 2;//current state, action, reward, next state

        m_memoryBuffer = new Queue<float>(capacity * m_experienceSize);
        randomIndexList = new int[capacity];

        for(int i = 0; i < capacity; ++i)
        {
            randomIndexList[i] = i;
        }
    }

    public void Add(float[] experience)
    {
        //array should be size of m_inputSize
        Debug.Assert(experience.Length == m_experienceSize);

        for(int i = 0; i < experience.Length; ++i)
        {
            m_memoryBuffer.Enqueue(experience[i]);
        }
    }

    public float[] GetSamples(int sampleSize)
    {
        float[] allSamples = m_memoryBuffer.ToArray();

        //if requested sample size is greater than the current memory size
        if(sampleSize > (m_memoryBuffer.Count / m_experienceSize))
        {
            return allSamples;
        }

        ShuffleClass.Shuffle<int>(randomIndexList);

        List<float> samples = new List<float>(sampleSize * m_experienceSize);

        for(int i = 0; i < sampleSize; ++i)
        {
            for(int j = 0; j < m_experienceSize; ++j)
            {
                int index = randomIndexList[i] * m_experienceSize;
                index += j;
                samples.Add(allSamples[index]);
            }
        }

        return samples.ToArray();
    }

    private Queue<float> m_memoryBuffer;
    private int m_experienceSize;
    private int[] randomIndexList;
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
