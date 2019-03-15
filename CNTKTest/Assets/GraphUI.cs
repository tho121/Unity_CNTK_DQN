using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

//credit: https://www.youtube.com/watch?v=CmU5-v-v1Qo
public class GraphUI : MonoBehaviour {

    public RectTransform graphContainer;
    public Sprite circleSprite;
	// Use this for initialization

    public void ShowGraph(List<float> values)
    {
        float graphHeight = graphContainer.rect.height;

        float yMax = Mathf.NegativeInfinity;

        for(int i = 0; i < values.Count; ++i)
        {
            if (values[i] > yMax)
            {
                yMax = values[i];
            }
        }

        float xSpacing = graphContainer.rect.width / values.Count;

        for(int i = 0; i < values.Count; ++i)
        {
            float x = i * xSpacing;
            float y = values[i] / yMax * graphHeight;
            Vector2 pos = new Vector2(x, y);

            if (i < m_circleList.Count)
            {
                m_circleList[i].anchoredPosition = pos;
            }
            else
            {
                CreateCircle(pos);
            }
        }
    }

    private void CreateCircle(Vector2 anchoredPosition)
    {
        GameObject go = new GameObject("circle", typeof(Image));
        go.transform.SetParent(graphContainer, false);
        go.GetComponent<Image>().sprite = circleSprite;

        RectTransform rt = go.GetComponent<RectTransform>();
        rt.anchoredPosition = anchoredPosition;
        rt.sizeDelta = new Vector2(4, 4);
        rt.anchorMax = new Vector2(0, 0);
        rt.anchorMin = new Vector2(0, 0);

        m_circleList.Add(rt);
    }

    private List<RectTransform> m_circleList = new List<RectTransform>();
}
