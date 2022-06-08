#define IMG_SIZE 784    // 784 = 28 * 25 / 이미지 픽셀 수 = 입력 레이어의 노드 수
#define IMG_COUNT 50000 // 이미지 개수
#define DIGIT_COUNT 10  // 출력 레이어 노드 수. 여기서는 신경망이 결론으로 낼 가지수인 0부터 9까지 총10개.

void recognition(float * images, float * network, int depth, int size, int * labels, float * confidences);

