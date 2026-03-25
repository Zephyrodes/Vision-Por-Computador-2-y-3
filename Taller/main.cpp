#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// =========================
// FACTORY DETECTOR
// =========================
Ptr<Feature2D> createDetector(const string& name) {
    if (name == "SIFT") return SIFT::create();
    if (name == "SURF") return SURF::create(400);
    if (name == "ORB")  return ORB::create();
    if (name == "FAST") return FastFeatureDetector::create(50);
    if (name == "BRISK") return BRISK::create();
    return nullptr;
}

// =========================
// FACTORY DESCRIPTOR
// =========================
Ptr<Feature2D> createDescriptor(const string& name) {
    if (name == "SIFT") return SIFT::create();
    if (name == "SURF") return SURF::create(400);
    if (name == "ORB")  return ORB::create();
    if (name == "BRIEF") return BriefDescriptorExtractor::create();
    if (name == "FREAK") return FREAK::create();
    if (name == "BRISK") return BRISK::create();
    return nullptr;
}

int main() {

    string img1_path = "../Data/box.png";
    string img2_path = "../Data/box_in_scene.png";

    Mat img1 = imread(img1_path, IMREAD_GRAYSCALE);
    Mat img2 = imread(img2_path, IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cout << "Error cargando imágenes" << endl;
        return -1;
    }

    vector<string> detectors = {"SIFT", "SURF", "ORB", "FAST", "BRISK"};
    vector<string> descriptors = {"SIFT", "SURF", "ORB", "BRIEF", "FREAK", "BRISK"};
    vector<string> matchers = {"BF", "FLANN"};

    ofstream file("resultados.csv");
    file << "Detector,Descriptor,Matcher,KP1,KP2,Matches,GoodMatches,Inliers,Tiempo,Estado\n";

    for (auto detName : detectors) {
        for (auto descName : descriptors) {
            for (auto matcherType : matchers) {

                bool valido = true;
                string estado = "OK";

                vector<KeyPoint> kp1, kp2;
                Mat des1, des2;
                vector<vector<DMatch>> knn_matches;
                vector<DMatch> good_matches;
                int inliers = 0;
                double t = 0;

                try {

                    Ptr<Feature2D> detector = createDetector(detName);
                    Ptr<Feature2D> descriptor = createDescriptor(descName);

                    if (!detector || !descriptor) {
                        throw runtime_error("No se pudo crear detector/descriptor");
                    }

                    // =========================
                    // VALIDACIONES TEÓRICAS
                    // =========================
                    if (detName == "FAST" && (descName == "SIFT" || descName == "SURF")) {
                        throw runtime_error("FAST no soporta SIFT/SURF");
                    }

                    bool binary = (descName == "ORB" || descName == "BRIEF" ||
                                   descName == "FREAK" || descName == "BRISK");

                    if (matcherType == "FLANN" && binary) {
                        throw runtime_error("FLANN no soporta descriptores binarios");
                    }

                    // =========================
                    // DETECCIÓN
                    // =========================
                    detector->detect(img1, kp1);
                    detector->detect(img2, kp2);

                    if (kp1.size() > 1000) kp1.resize(1000);
                    if (kp2.size() > 1000) kp2.resize(1000);

                    // =========================
                    // DESCRIPCIÓN
                    // =========================
                    descriptor->compute(img1, kp1, des1);
                    descriptor->compute(img2, kp2, des2);

                    if (des1.empty() || des2.empty())
                        throw runtime_error("Descriptores vacíos");

                    if (des1.rows < 10 || des2.rows < 10)
                        throw runtime_error("Muy pocos descriptores");

                    if (des1.cols != des2.cols)
                        throw runtime_error("Dimensiones incompatibles");

                    if (des1.type() != des2.type())
                        throw runtime_error("Tipos distintos");

                    Ptr<DescriptorMatcher> matcher;

                    // =========================
                    // MATCHING
                    // =========================
                    if (matcherType == "FLANN") {

                        if (des1.type() != CV_32F) des1.convertTo(des1, CV_32F);
                        if (des2.type() != CV_32F) des2.convertTo(des2, CV_32F);

                        matcher = makePtr<FlannBasedMatcher>(
                            makePtr<flann::KDTreeIndexParams>(5)
                        );

                    } else {
                        int normType = binary ? NORM_HAMMING : NORM_L2;
                        matcher = makePtr<BFMatcher>(normType);
                    }

                    double t0 = (double)getTickCount();

                    matcher->knnMatch(des1, des2, knn_matches, 2);

                    for (auto& m : knn_matches) {
                        if (m.size() == 2 && m[0].distance < 0.75 * m[1].distance) {
                            good_matches.push_back(m[0]);
                        }
                    }

                    // =========================
                    // HOMOGRAFÍA
                    // =========================
                    if (good_matches.size() >= 4) {

                        vector<Point2f> pts1, pts2;

                        for (auto& m : good_matches) {
                            pts1.push_back(kp1[m.queryIdx].pt);
                            pts2.push_back(kp2[m.trainIdx].pt);
                        }

                        Mat mask;
                        findHomography(pts1, pts2, RANSAC, 3.0, mask);

                        if (!mask.empty()) {
                            for (int i = 0; i < mask.rows; i++) {
                                if (mask.at<uchar>(i)) inliers++;
                            }
                        }
                    }

                    t = ((double)getTickCount() - t0) / getTickFrequency();

                }
                catch (const cv::Exception& e) {
                    valido = false;
                    estado = e.what();
                }
                catch (const exception& e) {
                    valido = false;
                    estado = e.what();
                }

                // =========================
                // GUARDAR RESULTADO
                // =========================
                file << detName << ","
                     << descName << ","
                     << matcherType << ","
                     << (valido ? kp1.size() : 0) << ","
                     << (valido ? kp2.size() : 0) << ","
                     << (valido ? knn_matches.size() : 0) << ","
                     << (valido ? good_matches.size() : 0) << ","
                     << (valido ? inliers : 0) << ","
                     << (valido ? t : 0) << ","
                     << estado << "\n";

                cout << detName << "+" << descName << "+" << matcherType
                     << " -> " << estado << endl;
            }
        }
    }

    file.close();
    cout << "\nResultados guardados en resultados.csv\n";

    return 0;
}
