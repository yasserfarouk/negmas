package genius.core;

import javax.swing.JTextArea;

/**
 *
 * @author Dmytro Tykhonov
 */
public class Logger {
    private JTextArea output;
    /** Creates a new instance of Logger */
    public Logger(JTextArea output) {
        this.output = output;
    }
    public synchronized void add(String text) {
        if(text!=null) {
            output.append("\n"+text);
            output.setCaretPosition(output.getDocument().getLength());
        }
     
    }
}
