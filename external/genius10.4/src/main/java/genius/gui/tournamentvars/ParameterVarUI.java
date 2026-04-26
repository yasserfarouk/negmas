package genius.gui.tournamentvars;

import java.awt.Panel;
import java.awt.Frame;
import java.awt.BorderLayout;
import java.util.ArrayList;
import java.util.Vector;

import javax.swing.JComboBox;

import genius.core.AgentParam;
import genius.gui.panels.DefaultOKCancelDialog;
/**
 * this shows a dialog where the user can select the parameter name 
 * @author wouter
 *
 */

public class ParameterVarUI extends DefaultOKCancelDialog {

	private static final long serialVersionUID = -6545959412141722703L;
	Panel panel;
	JComboBox combobox;
	
	ParameterVarUI(Frame owner,ArrayList<AgentParam> selectableparameters) throws Exception {
		super(owner, "Agent Parameter Selector");
		if (selectableparameters==null || selectableparameters.size()==0)
			throw new IllegalArgumentException("There are no selectable parameters because there are no selectable agents with parameters");
		panel=new Panel(new BorderLayout());
		combobox=new JComboBox(new Vector<AgentParam>(selectableparameters));
		panel.add(combobox,BorderLayout.CENTER);
	}
	
	public Panel getPanel() {
		return panel;
	}
	
	public Object ok() {
		return combobox.getSelectedItem();
	}
}