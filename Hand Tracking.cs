// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;

// public class HandTracking : MonoBehaviour
// {
//     // Start is called before the first frame update
//     public UDPReceive udpReceive;
//     public GameObject[] handPoints;
//     void Start()
//     {
        
//     }

//     // Update is called once per frame
//     void Update()
//     {
//         string data = udpReceive.data;

//         data = data.Remove(0, 1);
//         data = data.Remove(data.Length-1, 1);
//         print(data);
//         string[] points = data.Split(',');
//         print(points[0]);

//         //0        1*3      2*3
//         //x1,y1,z1,x2,y2,z2,x3,y3,z3

//         for ( int i = 0; i<21; i++)
//         {

//             float x = 7-float.Parse(points[i * 3])/100;
//             float y = float.Parse(points[i * 3 + 1]) / 100;
//             float z = float.Parse(points[i * 3 + 2]) / 100;

//             handPoints[i].transform.localPosition = new Vector3(x, y, z);

//         }


//     }
// }
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HandTracking : MonoBehaviour
{
    public UDPReceive udpReceive;
    public GameObject[] handPoints;   // 21 個手部關節球
    public GameObject targetObject;   // 你想要跟著旋轉的物體

    void Update()
    {
        string data = udpReceive.data;
        if (string.IsNullOrEmpty(data)) return; // 防呆：沒資料就跳過

        // 清理字串格式，例如 "[1.23, 100,200,...]" → "1.23,100,200,..."
        data = data.Trim('[', ']');
        string[] points = data.Split(',');

        if (points.Length < 64) return;  // 防呆：1 (yaw) + 21*3 (Landmarks) = 64

        // 1️⃣ 解析 yaw_deg 並旋轉物體
        float yawDeg;
        if (float.TryParse(points[0], out yawDeg))
        {
            // 注意：這裡加了「-」是因為 Python 與 Unity 座標系相反
            targetObject.transform.localRotation = Quaternion.Euler(0, -yawDeg, 0);
        }

        // 2️⃣ Landmark index 從 1 開始（因為 0 是 yaw_deg）
        for (int i = 0; i < 21; i++)
        {
            int baseIndex = 1 + i * 3;
            if (baseIndex + 2 >= points.Length) continue; // 保護界限

            float x = 7 - float.Parse(points[baseIndex]) / 100;
            float y = float.Parse(points[baseIndex + 1]) / 100;
            float z = float.Parse(points[baseIndex + 2]) / 100;

            handPoints[i].transform.localPosition = new Vector3(x, y, z);
        }
    }
}
