#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

// �� ���� ����ü ����
struct TrackedFace {
    cv::Rect rect;
    std::chrono::steady_clock::time_point firstSeen;
    std::chrono::steady_clock::time_point lastSeen;
};

int main() {
    // ��ķ�� ���� ���� VideoCapture ��ü�� �����մϴ� (�⺻������ 0�� ī�޶� ���ϴ�).
    cv::VideoCapture cap(0);

    // ī�޶� ������ �ʴ� ��� ���� �޽����� ����ϰ� �����մϴ�.
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // ��ķ �Ӽ��� ����մϴ�.
    std::cout << "Frame width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Frame height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    // Haar Cascade �з��⸦ �ε��մϴ�.
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: Could not load face cascade." << std::endl;
        return -1;
    }

    // �������� ������ Mat ��ü�� �����մϴ�.
    cv::Mat frame;
    std::vector<cv::Rect> faces;
    std::vector<TrackedFace> trackedFaces;

    // ���� ������ ���� �ǽð� ������ ����ؼ� ĸó�ϰ� ǥ���մϴ�.
    while (true) {
        // ��ķ���� �������� �о�ɴϴ�.
        cap >> frame;

        // �������� ��� �ִ��� Ȯ���ϰ�, ��� �ִٸ� ������ �����մϴ�.
        if (frame.empty()) {
            std::cerr << "Error: Could not read frame." << std::endl;
            break;
        }

        // �������� �׷��̽����Ϸ� ��ȯ�մϴ�.
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // ���� �����մϴ�.
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        auto now = std::chrono::steady_clock::now();
        for (const auto& face : faces) {
            bool found = false;
            for (auto& trackedFace : trackedFaces) {
                // ���� ���� ���� �󱼰��� ���缺�� Ȯ���մϴ�.
                if ((std::abs(face.x - trackedFace.rect.x) < 50) &&
                    (std::abs(face.y - trackedFace.rect.y) < 50) &&
                    (std::abs(face.width - trackedFace.rect.width) < 50) &&
                    (std::abs(face.height - trackedFace.rect.height) < 50)) {
                    trackedFace.rect = face;
                    trackedFace.lastSeen = now;
                    found = true;
                    break;
                }
            }
            // ���ο� ���̶�� �߰��մϴ�.
            if (!found) {
                trackedFaces.push_back({ face, now, now });
            }
        }

        // ������ ������ �����մϴ�.
        trackedFaces.erase(std::remove_if(trackedFaces.begin(), trackedFaces.end(), [now](const TrackedFace& trackedFace) {
            return std::chrono::duration_cast<std::chrono::seconds>(now - trackedFace.lastSeen).count() > 1;
            }), trackedFaces.end());

        // ������ �󱼿� �簢�� �������� �׸��ϴ�.
        for (const auto& trackedFace : trackedFaces) {
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - trackedFace.firstSeen).count();
            if (duration >= 10) {
                cv::rectangle(frame, trackedFace.rect, cv::Scalar(0, 0, 255), 2); // ������ ������
                cv::imshow("Webcam Live with Face Detection", frame);
                cv::waitKey(2000); // 2�� ���
                cv::Mat capturedFace = frame(trackedFace.rect).clone(); // �� �κи� �߶󳻱�
                cv::imwrite("captured_face.png", capturedFace);
                std::cout << "Captured face image saved as captured_face.png" << std::endl;
                cap.release();
                cv::destroyAllWindows();
                return 0; // ���α׷� ����
            }
            else {
                cv::rectangle(frame, trackedFace.rect, cv::Scalar(255, 0, 0), 2); // �Ķ��� ������
            }
        }

        // �������� â�� ǥ���մϴ�.
        cv::imshow("Webcam Live with Face Detection", frame);

        // 'q' Ű�� ������ ������ �����մϴ�.
        if (cv::waitKey(10) == 'q') {
            break;
        }
    }

    // ��ķ�� ��� â�� �����մϴ�.
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
