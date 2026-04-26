package genius.gui.uncertainty;
import javax.swing.BoxLayout;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class SteppingSliderExample
{
    
    public static void createAndShowGUI()
    {
        JFrame frame = new JFrame("SteppingSlider");
        frame.setSize(500, 120);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        final SteppingSlider steppingSlider = new SteppingSlider(10, 20, 50, 100);
        final String labelPrefix = "Slider value: ";
        final JLabel output = new JLabel(labelPrefix + steppingSlider.getDomainValue());
        steppingSlider.addChangeListener(new ChangeListener()
        {           
            @Override
            public void stateChanged(ChangeEvent evt)
            {
                output.setText(labelPrefix + steppingSlider.getDomainValue());
            }
        });
        frame.getContentPane().setLayout(
                new BoxLayout(frame.getContentPane(), 
                        BoxLayout.Y_AXIS));     
        frame.getContentPane().add(steppingSlider);
        frame.getContentPane().add(output);
        frame.setVisible(true);
    }

    public static void main(String[] args) throws Exception
    {
        SwingUtilities.invokeLater(new Runnable()
        {
            public void run()
            {
                createAndShowGUI();
            }
        });
    }
}