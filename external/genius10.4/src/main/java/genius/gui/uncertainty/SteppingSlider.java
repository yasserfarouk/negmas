package genius.gui.uncertainty;

import java.util.Arrays;
import java.util.Hashtable;

import javax.swing.JLabel;
import javax.swing.JSlider;

public class SteppingSlider extends JSlider
{
    private static final long serialVersionUID = -1195270044097152629L;
    private Integer[] values = { 10, 30, 60, 100 };
    private final Hashtable<Integer, JLabel> LABELS = new Hashtable<>();
    

    public SteppingSlider(Integer...allvalues)
    {
    	super(0, allvalues.length - 1, 0);
    	values = allvalues;
    	for(int i = 0; i < values.length; ++i)
        {
            LABELS.put(i, new JLabel(values[i].toString()));
        }
        setLabelTable(LABELS);      
        setPaintTicks(true);
        setPaintLabels(true);
        setSnapToTicks(true);
        setMajorTickSpacing(1);
    }

    public int getDomainValue()
    {
        return values[getValue()];
    }
    
    public void setDomainValue(int val)
    {
    	int binarySearch = Arrays.binarySearch(values, val);
    	int index = binarySearch >= 0 ? binarySearch : -binarySearch - 1; // insertion point
    	setValue(index);
    }
}
