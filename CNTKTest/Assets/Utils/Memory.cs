using System.Collections;
using System.Collections.Generic;
using UnityEngine;

class Memory
{
    public Memory(int experienceSize, int capacity = 1024)
    {
        m_experienceSize = experienceSize;
        m_capacity = capacity;
        m_memoryBuffer = new Queue<float>(capacity * m_experienceSize);
        m_randomIndexList = new int[capacity];

        for (int i = 0; i < capacity; ++i)
        {
            m_randomIndexList[i] = i;
        }
    }

    public void Add(float[] experience)
    {
        //array should be size of m_inputSize
        Debug.Assert(experience.Length == m_experienceSize);

        if(m_memoryBuffer.Count + experience.Length > m_capacity * m_experienceSize)
        {
            for (int i = 0; i < experience.Length; ++i)
            {
                m_memoryBuffer.Dequeue();
            }

            Debug.LogWarning("MEMORY DEQUEUE");
        }

        for (int i = 0; i < experience.Length; ++i)
        {
            m_memoryBuffer.Enqueue(experience[i]);
        }
    }

    public float[] GetAllSamples()
    {
        return GetSamples(m_memoryBuffer.Count / m_experienceSize);
    }

    public float[] GetSamples(int sampleSize)
    {
        float[] allSamples = m_memoryBuffer.ToArray();

        //if requested sample size is greater than the current memory size
        var currentMemoryCount = m_memoryBuffer.Count / m_experienceSize;
        if (sampleSize > currentMemoryCount)
        {
            return allSamples;
        }

        Utils.Shuffle<int>(m_randomIndexList);

        List<float> samples = new List<float>(sampleSize * m_experienceSize);

        int randomIndex = 0;
        for (int i = 0; i < sampleSize; ++i)
        {
            while (randomIndex < m_randomIndexList.Length && m_randomIndexList[randomIndex] >= currentMemoryCount)
            {
                randomIndex++;
            }

            int index = m_randomIndexList[randomIndex] * m_experienceSize;

            for (int j = 0; j < m_experienceSize; ++j)
            {
                samples.Add(allSamples[index + j]);
            }

            randomIndex++;
        }

        return samples.ToArray();
    }

    public void ClearMemory()
    {
        m_memoryBuffer.Clear();
    }

    public int GetExperienceSize()
    {
        return m_experienceSize;
    }

    public int GetCurrentMemorySize()
    {
        return m_memoryBuffer.Count / m_experienceSize;
    }

    private Queue<float> m_memoryBuffer;
    private int m_experienceSize;
    private int[] m_randomIndexList;
    private int m_capacity;
}
