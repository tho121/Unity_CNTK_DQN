using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallEnvironment : MonoBehaviour {

    public Transform platform;
    public Transform ball;

    public float MaxAngle = 40.0f;

    public bool selfReseting = false;

    //public Vector3 testRot;

	// Use this for initialization
	void Awake () {

        m_ballStartPos = ball.position;
        m_platformRB = platform.GetComponent<Rigidbody>();
        m_ballRB = ball.GetComponent<Rigidbody>();

        Reset();
    }

    // Update is called once per frame
    void Update () {
		
	}

    private void FixedUpdate()
    {
        if(selfReseting && IsDone())
        {
            Reset();
        }

        //Act(testRot.x, testRot.z);
    }

    public void Reset()
    {
        platform.rotation = Quaternion.Euler(
            Random.Range(-MaxAngle, MaxAngle),
            0.0f,
            Random.Range(-MaxAngle, MaxAngle));

        m_ballRB.velocity = Vector3.zero;
        ball.position = m_ballStartPos;
    }

    public bool IsDone()
    {
        //return ball.transform.position.y < -0.5f || ball.transform.position.y > 1.5f;
        return (ball.transform.position - platform.transform.position).sqrMagnitude > 10.0f;
    }

    public float[] GetState()
    {
        m_state[0] = ball.position.x;
        m_state[1] = ball.position.y;
        m_state[2] = ball.position.z;

        var euler = platform.rotation.eulerAngles;

        m_state[3] = euler.x / 360.0f;
        m_state[4] = euler.y / 360.0f;
        m_state[5] = euler.z / 360.0f;

        m_state[6] = m_ballRB.velocity.x;
        m_state[7] = m_ballRB.velocity.y;
        m_state[8] = m_ballRB.velocity.z;

        return m_state;
    }

    public int GetStateSize()
    {
        return m_state.Length;
    }

    public int GetActionSize()
    {
        return 2;   //x rotation and z rotation
    }

    //values from -1 to 1
    public void Act(float rotX, float rotZ)
    {


        //Debug.Log(m_platformRB.rotation.eulerAngles);
        //var result = Quaternion.Inverse(m_platformRB.rotation) * Quaternion.Euler(rotX * MaxAngle, 0.0f, rotZ * MaxAngle);
        //m_platformRB.MoveRotation(result);


        var result = Quaternion.RotateTowards(platform.rotation, Quaternion.Euler(rotX * MaxAngle, 0.0f, rotZ * MaxAngle), MaxAngle * Time.fixedDeltaTime * 3.0f);
        m_platformRB.MoveRotation(result);
        //m_platformRB.rotation = Quaternion.Euler(rotX * MaxAngle, 0.0f, rotZ * MaxAngle);
    }

    private Vector3 m_ballStartPos;
    private Rigidbody m_platformRB;
    private Rigidbody m_ballRB;
    private float[] m_state = new float[9];
}
