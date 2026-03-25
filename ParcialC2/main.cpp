#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void ejecutarPipeline(string metodo,
                      Ptr<Feature2D> detector,
                      bool usarFREAK = false)
{
    string face_path = "../Data/haarcascades/haarcascade_frontalface_alt.xml";
    string eyes_path = "../Data/haarcascades/haarcascade_eye.xml";
    string video_path = "../Data/blais.mp4";
    string img_path   = "../Data/book.png";

    CascadeClassifier face_cascade, eyes_cascade;
    face_cascade.load(face_path);
    eyes_cascade.load(eyes_path);

    // ==============================
    // IMAGEN BASE (book.png)
    // ==============================
    Mat img = imread(img_path);
    if (img.empty()) {
        cerr << "❌ Error cargando book.png\n";
        return;
    }

    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);

    vector<Rect> faces_img;
    face_cascade.detectMultiScale(gray_img, faces_img, 1.1, 3);

    if (faces_img.empty()) {
        cerr << "❌ No se detectó rostro en book.png\n";
        return;
    }

    // ==============================
    // ROI REAL (SOLO LA CARA)
    // ==============================
    Rect face_roi = faces_img[0];

    Mat roi_color = img(face_roi);
    Mat roi_gray;
    cvtColor(roi_color, roi_gray, COLOR_BGR2GRAY);

    // ==============================
    // VENTANA 2 (SOLO ROI)
    // ==============================
    imshow("Objeto detectado ROIs", roi_color);

    // ==============================
    // FEATURES ROI
    // ==============================
    vector<KeyPoint> kp_obj;
    Mat desc_obj;

    Ptr<FREAK> freak;
    Ptr<FastFeatureDetector> fast;

    if (usarFREAK)
    {
        fast = FastFeatureDetector::create();
        freak = FREAK::create();

        fast->detect(roi_gray, kp_obj);
        freak->compute(roi_gray, kp_obj, desc_obj);
    }
    else
    {
        detector->detectAndCompute(roi_gray, noArray(), kp_obj, desc_obj);
    }

    // ==============================
    // VIDEO
    // ==============================
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "❌ Error abriendo video\n";
        return;
    }

    BFMatcher matcher(NORM_HAMMING);

    Mat frame, gray;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // ==============================
        // VENTANA 1 (VIDEO + DETECCIÓN)
        // ==============================
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3);

        Mat frame_draw = frame.clone();

        Rect face;
        if (!faces.empty())
        {
            face = faces[0];

            Point center(face.x + face.width/2,
                         face.y + face.height/2);

            // cara fucsia
            circle(frame_draw, center, face.width/2,
                   Scalar(255,0,255), 3);

            // ojos azul
            Mat faceROI_gray = gray(face);
            vector<Rect> eyes;
            eyes_cascade.detectMultiScale(faceROI_gray, eyes);

            for (auto &e : eyes)
            {
                Point eye_center(face.x + e.x + e.width/2,
                                 face.y + e.y + e.height/2);

                circle(frame_draw, eye_center, 10,
                       Scalar(255,0,0), 2);
            }
        }

        imshow("Capture - Objeto detectado", frame_draw);

        // ==============================
        // FEATURES ESCENA
        // ==============================
        vector<KeyPoint> kp_scene;
        Mat desc_scene;

        if (usarFREAK)
        {
            fast->detect(gray, kp_scene);
            freak->compute(gray, kp_scene, desc_scene);
        }
        else
        {
            detector->detectAndCompute(gray, noArray(),
                                       kp_scene, desc_scene);
        }

        if (desc_scene.empty() || desc_obj.empty()) continue;

        // ==============================
        // VENTANA 3 (MATCHES GENERALES)
        // ==============================
        vector<DMatch> matches;
        matcher.match(desc_obj, desc_scene, matches);

        Mat img_matches;
        drawMatches(roi_color, kp_obj,
                    frame, kp_scene,
                    matches, img_matches);

        imshow(metodo + " Matches", img_matches);

        // ==============================
        // FILTRAR MATCHES EN ROSTRO
        // ==============================
        vector<DMatch> filtered_matches;

        if (!faces.empty())
        {
            for (auto &m : matches)
            {
                Point2f pt = kp_scene[m.trainIdx].pt;
                if (face.contains(pt))
                    filtered_matches.push_back(m);
            }
        }

        // ==============================
        // HOMOGRAFIA + CUADRO VERDE
        // ==============================
        Mat frame_final = frame.clone();

        if (filtered_matches.size() >= 4)
        {
            vector<Point2f> obj, scene;

            for (auto &m : filtered_matches)
            {
                obj.push_back(kp_obj[m.queryIdx].pt);
                scene.push_back(kp_scene[m.trainIdx].pt);
            }

            Mat H = findHomography(obj, scene, RANSAC);

            if (!H.empty())
            {
                vector<Point2f> corners(4);
                corners[0] = Point(0,0);
                corners[1] = Point(roi_color.cols,0);
                corners[2] = Point(roi_color.cols,roi_color.rows);
                corners[3] = Point(0,roi_color.rows);

                vector<Point2f> scene_corners;
                perspectiveTransform(corners, scene_corners, H);

                for (int i=0;i<4;i++)
                {
                    line(frame_final,
                         scene_corners[i],
                         scene_corners[(i+1)%4],
                         Scalar(0,255,0), 3);
                }
            }
        }

        // ==============================
        // VENTANA 4 (MATCHES FILTRADOS)
        // ==============================
        Mat img_final;
        drawMatches(roi_color, kp_obj,
                    frame_final, kp_scene,
                    filtered_matches,
                    img_final);

        imshow("Matches Finales en el objeto detectado", img_final);

        if (waitKey(30) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

int main()
{
    Ptr<BRISK> brisk = BRISK::create();

    ejecutarPipeline("BRISK", brisk, false);
    ejecutarPipeline("FREAK", nullptr, true);

    return 0;
}
