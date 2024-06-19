#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

// 얼굴 정보 구조체 정의
struct TrackedFace {
    cv::Rect rect;
    std::chrono::steady_clock::time_point firstSeen;
    std::chrono::steady_clock::time_point lastSeen;
};

int main() {
    // 웹캠을 열기 위해 VideoCapture 객체를 생성합니다 (기본적으로 0번 카메라를 엽니다).
    cv::VideoCapture cap(0);

    // 카메라가 열리지 않는 경우 오류 메시지를 출력하고 종료합니다.
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // 웹캠 속성을 출력합니다.
    std::cout << "Frame width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Frame height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    // Haar Cascade 분류기를 로드합니다.
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: Could not load face cascade." << std::endl;
        return -1;
    }

    // 프레임을 저장할 Mat 객체를 생성합니다.
    cv::Mat frame;
    std::vector<cv::Rect> faces;
    std::vector<TrackedFace> trackedFaces;

    // 무한 루프를 통해 실시간 영상을 계속해서 캡처하고 표시합니다.
    while (true) {
        // 웹캠에서 프레임을 읽어옵니다.
        cap >> frame;

        // 프레임이 비어 있는지 확인하고, 비어 있다면 루프를 종료합니다.
        if (frame.empty()) {
            std::cerr << "Error: Could not read frame." << std::endl;
            break;
        }

        // 프레임을 그레이스케일로 변환합니다.
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // 얼굴을 감지합니다.
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        auto now = std::chrono::steady_clock::now();
        for (const auto& face : faces) {
            bool found = false;
            for (auto& trackedFace : trackedFaces) {
                // 기존 추적 중인 얼굴과의 유사성을 확인합니다.
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
            // 새로운 얼굴이라면 추가합니다.
            if (!found) {
                trackedFaces.push_back({ face, now, now });
            }
        }

        // 오래된 추적을 제거합니다.
        trackedFaces.erase(std::remove_if(trackedFaces.begin(), trackedFaces.end(), [now](const TrackedFace& trackedFace) {
            return std::chrono::duration_cast<std::chrono::seconds>(now - trackedFace.lastSeen).count() > 1;
            }), trackedFaces.end());

        // 감지된 얼굴에 사각형 프레임을 그립니다.
        for (const auto& trackedFace : trackedFaces) {
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - trackedFace.firstSeen).count();
            if (duration >= 10) {
                cv::rectangle(frame, trackedFace.rect, cv::Scalar(0, 0, 255), 2); // 빨간색 프레임
                cv::imshow("Webcam Live with Face Detection", frame);
                cv::waitKey(2000); // 2초 대기
                cv::Mat capturedFace = frame(trackedFace.rect).clone(); // 얼굴 부분만 잘라내기
                cv::imwrite("captured_face.png", capturedFace);
                std::cout << "Captured face image saved as captured_face.png" << std::endl;
                cap.release();
                cv::destroyAllWindows();
                return 0; // 프로그램 종료
            }
            else {
                cv::rectangle(frame, trackedFace.rect, cv::Scalar(255, 0, 0), 2); // 파란색 프레임
            }
        }

        // 프레임을 창에 표시합니다.
        cv::imshow("Webcam Live with Face Detection", frame);

        // 'q' 키를 누르면 루프를 종료합니다.
        if (cv::waitKey(10) == 'q') {
            break;
        }
    }

    // 웹캠과 모든 창을 정리합니다.
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
