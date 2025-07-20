import train
import test

def main():
    print("CartPole DQN 학습 시작")
    train.main()  # train.py에 main() 함수가 있다고 가정
    # 테스트도 같이 하고 싶으면 아래 주석 해제
    # print("CartPole DQN 테스트 시작")
    # test.main()

if __name__ == "__main__":
    main()
