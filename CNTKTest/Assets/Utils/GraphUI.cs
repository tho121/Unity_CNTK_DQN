using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

//credit: https://www.youtube.com/watch?v=CmU5-v-v1Qo
public class GraphUI : MonoBehaviour {

    public RectTransform graphContainer;
    public Sprite circleSprite;
    public List<Color> colorIndex = new List<Color>();
	// Use this for initialization

    public void ShowGraph(List<float> values, int index = 0)
    {
        //graphContainer.gameObject.SetActive(true);

        if (m_graphs.Count <= index)
        {
            var graphGO = Instantiate(graphContainer.gameObject);
            graphGO.transform.SetParent(graphContainer.gameObject.transform.parent);

            m_graphs.Add(graphGO.GetComponent<RectTransform>());
            m_circleList.Add(index, new List<RectTransform>());
        }

        RectTransform graph = m_graphs[index];
        List<RectTransform> circles = m_circleList[index];

        float graphHeight = Screen.height; //graphContainer.rect.height;

        float yMax = Mathf.NegativeInfinity;
        float yMin = Mathf.Infinity;

        for(int i = 0; i < values.Count; ++i)
        {
            if (values[i] > yMax)
            {
                yMax = values[i];
            }

            if (values[i] < yMin)
            {
                yMin = values[i];
            }
        }

        Debug.Log(index + ": " + "Low: " + yMin + " High: " + yMax);

        float xSpacing = (float)Screen.width / values.Count; //graphContainer.rect.width / values.Count;

        for (int i = 0; i < values.Count; ++i)
        {
            float x = i * xSpacing;
            float y = Mathf.InverseLerp( yMin, yMax, values[i]) * graphHeight;
            Vector2 pos = new Vector2(x, y);

            if (i < circles.Count)
            {
                circles[i].anchoredPosition = pos;
            }
            else
            {
                CreateCircle(pos, colorIndex.Count > index ? colorIndex[index] : Color.white, index);
            }
        }
    }

    private void CreateCircle(Vector2 anchoredPosition, Color color, int graphIndex = 0)
    {
        GameObject go = new GameObject("circle", typeof(Image));
        go.transform.SetParent(m_graphs[graphIndex], false);
        var image = go.GetComponent<Image>();
        image.sprite = circleSprite;
        image.color = color;

        RectTransform rt = go.GetComponent<RectTransform>();
        rt.anchoredPosition = anchoredPosition;
        rt.sizeDelta = new Vector2(4, 4);
        rt.anchorMax = new Vector2(0, 0);
        rt.anchorMin = new Vector2(0, 0);

        m_circleList[graphIndex].Add(rt);
    }


    private List<RectTransform> m_graphs = new List<RectTransform>();
    private Dictionary<int, List<RectTransform>> m_circleList = new Dictionary<int, List<RectTransform>>();
}
