package genius.gui.tree;

import java.awt.*;
import java.awt.event.*;
import java.text.*;
import javax.swing.*;
import javax.swing.event.*;

import genius.core.issue.*;
import genius.core.utility.*;

/**
* @author Richard Noorlandt
* 
* Wouter: WeightSlider is a GUI component in the Tree,
* it is always there but may be invisible (if the objective/issue has no weight).
*/

public class WeightSlider extends JPanel implements ChangeListener, ItemListener {

	private static final long serialVersionUID = 4049454839954128143L;
	//Attributes
	static final int MIN_VALUE = 0;
	static final int MAX_VALUE = 1000;
	
	static final Color BACKGROUND = Color.white;
	
	//private Objective objective;
	//private Evaluator evaluator;
	private NegotiatorTreeTableModel tableModel;
	private Objective objective; // for which objective is this the weight slider
	private JCheckBox lock;
	private JSlider slider;
	private JFormattedTextField valueField;
	
	private double weight = 0; // < Might change, should probably be done from within objective, model or evaluator. Ony the question is: which?
	
	
	
	//Constructors
	
	//public WeightSlider(Objective obj) {
	public WeightSlider(NegotiatorTreeTableModel model, Objective obj) {
		tableModel = model;
		objective = obj;
		
		this.setBackground(BACKGROUND);
		this.setLayout(new FlowLayout());
		
		slider = new JSlider(MIN_VALUE, MAX_VALUE);
		//slider.setBackground(BACKGROUND);
		slider.setToolTipText("Drag to change the weight");
		slider.addChangeListener(this);
		this.add(slider);
		
		NumberFormat format = NumberFormat.getNumberInstance();
		format.setMaximumFractionDigits(2);
		valueField = new JFormattedTextField(format);
		valueField.setColumns(4);
		valueField.setToolTipText("Fill in a weight between 0 and 1");
		valueField.setEditable(false);
		this.add(valueField);
	
		lock = new JCheckBox();
		lock.setBackground(BACKGROUND);
		lock.setToolTipText("Lock weight");
		lock.addItemListener(this);
		this.add(lock);
		
		// Wouter: new code using the hasWeight field.
		boolean hasweight=true;
		AdditiveUtilitySpace us=tableModel.getUtilitySpace();
		if (us != null)
		{
			Evaluator ev=us.getEvaluator(obj.getNumber());
			//if (ev==null) System.out.println("no evaluator found for "+obj);
			if (ev instanceof EvaluatorObjective) // should always be the case?? 
				hasweight=((EvaluatorObjective)ev).getHasWeight();
		}
		setVisible(hasweight);

		
		//Added by Herbert
		if((tableModel.getUtilitySpace() != null) && tableModel.getUtilitySpace().getEvaluator(obj.getNumber())== null || obj.getName().equals("root")){
			slider.setVisible(false);
			valueField.setVisible(false);
			lock.setVisible(false);
		}
	
		updatePreferredSize();
		
	}
	
	
	//Methods	
	
	/**
	 * Sets this slider to be visible or invisible, dependent on whether there is an evaluator available. When there is no evaluator, the
	 * slider wil always be invisible.
	 * @param vis True makes the slider visible, false wil hide the slider.
	 */
	public void setVisible(boolean vis){
		if(((tableModel.getUtilitySpace() != null) && (tableModel.getUtilitySpace().getEvaluator(objective.getNumber()) == null)) || (objective.getName().equals("root"))){
			slider.setVisible(false);
			valueField.setVisible(false);
			lock.setVisible(false);
		}else{
			slider.setVisible(vis);
			valueField.setVisible(vis);
			lock.setVisible(vis);
		}
	}
	
	/**
	 * Converts an int between MIN_VALUE and MAX_VALUE to a double between 0 and 1. This method is
	 * necessary because JSlider only works on int, and weights are doubles between 0 and 1.
	 * @param value the value to be converted.
	 * @return the value converted to a double.
	 */
	private double convertToDouble(int value) {
		if (value < MIN_VALUE)
			return 0;
		if (value > MAX_VALUE)
			return 1;
		
		return (double)value / (double)( (MAX_VALUE) - MIN_VALUE);
	}
	
	/**
	 * Converts a double between 0 and 1 to an int between MIN_VALUE and MAX_VALUE. This method is
	 * necessary because JSlider only works on int, and weights are in doubles between 0 and 1.
	 * @param value the value to e converted.
	 * @return the value converted to an int between MIN_VALUE and MAX_VALUE.
	 */
	private int convertToInt(double value) {
		if (value < 0)
			return MIN_VALUE;
		if (value > 1)
			return MAX_VALUE;
		
		return (int)(value*((double)MAX_VALUE - (double)MIN_VALUE)) + MIN_VALUE;
	}
	
	/**
	 * 
	 * @return the weight.
	 */
	public double getWeight() {
		return weight;
	}
	
	/**
	 * 
	 * @param newWeight the new weight.
	 */
	public void setWeight(double newWeight) {
		weight = newWeight;
		valueField.setValue(weight);
		slider.setValue(convertToInt(weight));
		// Wouter: try to call explicit treeStructureChanged after change.
		tableModel.treeNodesChanged(objective, objective.getPath().getPath());
	}
	
	public Objective getObjective() {
		return objective;
	}
	
	//public void setObjective(Objective obj) {
	//
	//}
	
	/**
	 * Tries to set the new weight, and signals the NegotiatorTreeTableModel that weights are updated.
	 * @param newWeight the new weight.
	 */
	public void changeWeight(double newWeight) {
		weight = tableModel.getUtilitySpace().setWeight(objective, newWeight);
		tableModel.updateWeights(this, weight);
		setWeight(weight); //redundant

	}
	
	public void stateChanged(ChangeEvent e) {
		if (e.getSource() != slider){
			return;
		}
		double newWeight = convertToDouble(slider.getValue());
		valueField.setValue(newWeight);
		changeWeight(newWeight);
	}
	
	/**
	 * Implementation of ItemListener, which is registered on the checkbox. If the checkbox state changes,
	 * the slider and textfield will be locked or unlocked. An unchecked checkbox means that the weight
	 * can be changed.
	 * @param e as defined by the ItemListener interface.
	 */
	public void itemStateChanged(ItemEvent e) {
		if (e.getStateChange() == ItemEvent.SELECTED) {
			slider.setEnabled(false);
			valueField.setEnabled(false);
			tableModel.getUtilitySpace().lock(objective);
		}
		//Otherwise, it is deselected
		else {
			slider.setEnabled(true);
			valueField.setEnabled(true);
			tableModel.getUtilitySpace().unlock(objective);
		}
	}
	
	/**
	 * Calculates and sets this objects preferred size, based on its subcomponents.
	 *
	 */
	protected void updatePreferredSize() {
		int prefHeight = lock.getPreferredSize().height;
		if (slider.getPreferredSize().height > prefHeight)
			prefHeight = slider.getPreferredSize().height;
		if (valueField.getPreferredSize().height > prefHeight)
			prefHeight = valueField.getPreferredSize().height;
		
		int prefWidth = lock.getPreferredSize().width +  slider.getPreferredSize().width + valueField.getPreferredSize().width;
		
		this.setPreferredSize(new Dimension(prefWidth, prefHeight));
	}
	
	public void forceRedraw(){
		try{
		this.changeWeight(this.getWeight());
		}catch(Exception e){
			//do nothing, maybe we don't have a weight
		}
		this.setVisible(true);
		this.repaint();
	}
}

