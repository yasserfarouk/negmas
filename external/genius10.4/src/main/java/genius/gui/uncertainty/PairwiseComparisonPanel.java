package genius.gui.uncertainty;

import java.util.Hashtable;

import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import genius.gui.panels.ButtonPanel;
import genius.gui.panels.ExtendedCheckboxPanel;

public class PairwiseComparisonPanel extends JPanel
{	
	private static final long serialVersionUID = 7745677414933157895L;
	
	private PairwiseComparisonModel model;
	
	private final JLabel label;
	private final JLabel numberOfCompsLabel;
	private final JLabel rankingsLabel;
	
	private JPanel uncertaintyLevelPanel;
	private SteppingSlider uncertaintySlider;
	
	final JTextField text = new JTextField(4);
	
	private JSlider errorSlider;
	private JLabel errorLabel;
	
	private ExtendedCheckboxPanel experimentalBox;
	private final ButtonPanel okButton;


	public PairwiseComparisonPanel(PairwiseComparisonModel model) 
	{
		JFrame frame = new JFrame("Uncertainty options");
		this.model = model;

		this.label = new JLabel("Set user model parameters");
		this.numberOfCompsLabel = new JLabel("(Maximum number of rankings = " + this.model.getMaxNumberInComps() +")");
		
		this.uncertaintyLevelPanel = new JPanel(new GridBagLayout());
        
        GridBagConstraints c = new GridBagConstraints();
        c.insets = new Insets(10,10,10,10);
        
        
		this.uncertaintySlider = createUncertaintySlider();
		this.rankingsLabel = new JLabel("Amount of rankings");

		uncertaintySlider.addChangeListener(new ChangeListener(){
            @Override
            public void stateChanged(ChangeEvent e) 
            {
         	
            	// Don't set the text field if that was the one triggering this event
            	if (uncertaintySlider.isFocusOwner())
            		text.setText(String.valueOf(uncertaintySlider.getDomainValue()));
            	
            	model.setNumberOfComparisons(uncertaintySlider.getDomainValue());
            }
        });
		
        text.addKeyListener(new KeyAdapter(){
            @Override
            public void keyReleased(KeyEvent ke) {
                String typed = text.getText();
                if(!typed.matches("\\d+") || typed.length() > 3) {
                	uncertaintySlider.setDomainValue(0);
                    return;
                }
                int value = Integer.parseInt(typed);
                uncertaintySlider.setDomainValue(value);
				model.setNumberOfComparisons(value);	
            }
        });
        
        text.setMaximumSize(new Dimension(20, 30));

        // add components to the panel
        c.gridx = 0;
        c.gridy = 0;
        uncertaintyLevelPanel.add(rankingsLabel, c);
 
        c.gridx = 1;
        uncertaintyLevelPanel.add(uncertaintySlider, c);
         
        c.gridx = 2;
        uncertaintyLevelPanel.add(text, c);	
		
        c.gridx = 0;
        c.gridy = 1;
        c.weighty = 1.0;   //request any extra vertical space
		this.errorLabel = new JLabel("Error rate");
		uncertaintyLevelPanel.add(errorLabel, c);
		
		c.gridx = 1;
        c.gridy = 1;
		this.errorSlider = createErrorSlider();
		uncertaintyLevelPanel.add(errorSlider, c);

		c.gridx = 0;
        c.gridy = 2;
        c.gridwidth = 3;

		experimentalBox = new ExtendedCheckboxPanel("Grant agent access to real utility function (experimental setup)", 
				model.getExperimentalModel());
		experimentalBox.getCheckBox().setSelected(true);
		uncertaintyLevelPanel.add(experimentalBox, c);

		c.gridx = 1;
        c.gridy = 3;
        c.gridwidth = 1;
		
		okButton = new ButtonPanel("OK", model.getConfirmationModel());
		uncertaintyLevelPanel.add(okButton, c);
		
//		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		
		add(numberOfCompsLabel);
		add(uncertaintyLevelPanel);
		
		frame.add(this);
	    frame.setSize(500,300);
	    frame.setLocationRelativeTo(null);
		frame.setVisible(true);
	}
	
		
	private SteppingSlider createUncertaintySlider() 
	{
		final SteppingSlider steppingSlider = new SteppingSlider(model.getPossibleValues());
        return steppingSlider;
	}
	
	private JSlider createErrorSlider() {
		JSlider slider = new JSlider(JSlider.HORIZONTAL,0,100,0);
		slider.setMajorTickSpacing( 1 );		
		slider.setMinorTickSpacing( 2 );		
		Hashtable<Integer, JLabel> sliderLabelsTable = new Hashtable<Integer, JLabel>();
		sliderLabelsTable.put( 0 , new JLabel("0%") );
		sliderLabelsTable.put( 100 , new JLabel("100%") );
	    slider.setLabelTable(sliderLabelsTable);
	    slider.setPaintLabels(true);
//	    slider.setMaximumSize(new Dimension(50,50));
	    slider.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				model.getErrorModel().setText(String.valueOf((double) slider.getValue() / 100));
		}			
	});			
		return slider;
	}
	
	
}
