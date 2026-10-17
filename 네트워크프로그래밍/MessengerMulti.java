import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.*;
import java.net.*;

import javax.swing.JFrame;
import javax.swing.JTextArea;
import javax.swing.JTextField;

public class MessengerMulti {
    
    protected JTextField textField;
    protected JTextArea textArea;
    DataInputStream is;
    DataOutputStream os;
    Socket s;
    Thread receiveThread;
    volatile boolean running = true; // 스레드 종료 플래그

    public MessengerMulti() throws IOException {
        MyFrame f = new MyFrame();
        
        s = new Socket("localhost", 5000); 
        System.out.println("서버에 연결되었습니다.");
        
        is = new DataInputStream(s.getInputStream());
        os = new DataOutputStream(s.getOutputStream());

        receiveThread = new Thread(new Runnable() {
            @Override
            public void run() {
                while (running) {
                    try {
                        String msg = is.readUTF();
                        textArea.append("RECEIVED: " + msg + "\n");
                        textArea.setCaretPosition(textArea.getDocument().getLength());
                    } catch (IOException e) {
                        if (running) {
                            textArea.append("서버와의 연결이 끊어졌습니다.\n");
                        }
                        break;
                    }
                }
            }
        });
        receiveThread.start();
    }

    // 리소스 정리 메서드
    public void cleanup() {
        running = false;
        
        try {
            if (os != null) os.close();
            if (is != null) is.close();
            if (s != null && !s.isClosed()) s.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        System.out.println("리소스가 정리되었습니다.");
    }

    class MyFrame extends JFrame implements ActionListener {
    
        public MyFrame() {
            super("MessengerA");
            setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            // 창 닫을 때 리소스 정리
            addWindowListener(new WindowAdapter() {
                @Override
                public void windowClosing(WindowEvent e) {
                    System.out.println("프로그램을 종료합니다...");
                    cleanup();
                }
            });

            textField = new JTextField(30);
            textField.addActionListener(this);

            textArea = new JTextArea(10, 30);
            textArea.setEditable(false);

            add(textField, BorderLayout.PAGE_END);
            add(textArea, BorderLayout.CENTER);
            pack();
            setVisible(true);
        }

        public void actionPerformed(ActionEvent evt) {
            String s = textField.getText();
            
            if (s.isEmpty()) return; // 빈 메시지 전송 방지
            
            try {
                os.writeUTF(s);
                os.flush(); // 버퍼 비우기
                textArea.append("SENT: " + s + "\n");
            } catch (IOException e) {
                textArea.append("메시지 전송 실패: 서버 연결이 끊어졌습니다.\n");
                e.printStackTrace();
            }
            
            textField.selectAll();
            textArea.setCaretPosition(textArea.getDocument().getLength());
        }
    }

    public static void main(String[] args) throws IOException {
        MessengerMulti m = new MessengerMulti();
    }
}



//창(JFrame 등)의 동작(열기, 닫기, 최소화 등)을 감지할 때 WindowListener 인터페이스를 구현
//
//public interface WindowListener {
//    void windowOpened(WindowEvent e);
//    void windowClosing(WindowEvent e);
//    void windowClosed(WindowEvent e);
//    void windowIconified(WindowEvent e);
//    void windowDeiconified(WindowEvent e);
//    void windowActivated(WindowEvent e);
//    void windowDeactivated(WindowEvent e);
//}
//
//필요한 메서드만 오버라이드 가능하도록 만들어 놓은 것
//모든 메서드가 비어 있는(아무 일도 하지 않는) 상태로 미리 구현
//public abstract class WindowAdapter implements WindowListener {
//    public void windowOpened(WindowEvent e) {}
//    public void windowClosing(WindowEvent e) {}
//    public void windowClosed(WindowEvent e) {}
//    public void windowIconified(WindowEvent e) {}
//    public void windowDeiconified(WindowEvent e) {}
//    public void windowActivated(WindowEvent e) {}
//    public void windowDeactivated(WindowEvent e) {}
//}