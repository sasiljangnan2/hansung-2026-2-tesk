import java.io.IOException;
import java.io.InputStream;

public class TestKeyboard {
    public static void main(String[] args) {

        try {
            InputStream is = System.in;

            System.out.print("텍스트 입력: ");
            
            byte[] byteData = new byte[1024];
            int bytesRead = is.read(byteData);

            String inputMessage = new String(byteData, 0, bytesRead);
            System.out.print(inputMessage);

        } catch (IOException e) {
            System.out.println("오류가 발생했습니다.");
        }
    }
}


