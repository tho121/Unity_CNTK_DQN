//https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
//credit to this guy!

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SumTree
{
    public SumTree(int capacity)
    {
        m_capacity = capacity;
        m_tree = new List<float>(new float[capacity * 2 - 1]);
        m_data = new List<float[]>(capacity);

        for(int i = 0; i < capacity; ++i)
        {
            m_data.Add(new float[] { });
        }
    }

    public void Propagate(int idx, float change)
    {
        int parent = ((idx - 1) / 2);

        m_tree[parent] += change;

        if(parent != 0)
        {
            Propagate(parent, change);
        }
    }

    public int Retrieve(int idx, float s)
    {
        int left = 2 * idx + 1;
        int right = left + 1;

        if( left >= m_tree.Count)
        {
            return idx;
        }

        if(s <= m_tree[left])
        {
            return Retrieve(left, s);
        }
        else
        {
            return Retrieve(right, s - m_tree[left]);
        }
    }

    public float Total()
    {
        return m_tree[0];
    }

    public void Add(float p, float[] data)
    {
        int idx = m_write + m_capacity - 1;

        m_data[m_write] = data;

        Update(idx, p);

        m_write += 1;

        if(m_write >= m_capacity)
        {
            m_write = 0;
        }
    }

    public void Update(int idx, float p)
    {
        float change = p - m_tree[idx];
        m_tree[idx] = p;

        Propagate(idx, change);
    }

    public float[] Get(float s, out int idx, out float treeVal)
    {
        idx = Retrieve(0, s);

        int dataIdx = idx - m_capacity + 1;

        treeVal = m_tree[idx];

        if(dataIdx >= m_data.Count)
        {
            Debug.Log("wtf");
        }

        return m_data[dataIdx];
    }

    List<float> m_tree;
    List<float[]> m_data;
    int m_write = 0;
    int m_capacity = 0;
}
