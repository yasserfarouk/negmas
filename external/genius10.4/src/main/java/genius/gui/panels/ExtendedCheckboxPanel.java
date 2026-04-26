package genius.gui.panels;



import java.awt.BorderLayout;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;

import genius.core.listener.Listener;

/** Another implementation of the CheckBox Panel. Uses Swing's JCheckBox. Works only with mouse at the moment
 * @author dimtsi
 *
 */
public class ExtendedCheckboxPanel extends JPanel{

	private JLabel label;
	private JCheckBox checkBox;
	private BooleanModel model;

	/**
	 * 
	 * @param text
	 *            the text for the label
	 * @param boolModel
	 *            the boolean model
	 */
	public ExtendedCheckboxPanel(final String text, final BooleanModel boolModel) {
		this.model = boolModel;

		setLayout(new BorderLayout());
		label = new JLabel(text);
		checkBox = new JCheckBox();
		enable1();

		// connect the box to the model,

		add(label, BorderLayout.CENTER);
		add(checkBox, BorderLayout.EAST);
	
		checkBox.addMouseListener(new MouseListener() {

			@Override
			public void mouseClicked(MouseEvent arg0) {
					model.setValue(checkBox.isSelected());
			}
				

			@Override
			public void mouseEntered(MouseEvent arg0) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void mouseExited(MouseEvent arg0) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void mousePressed(MouseEvent arg0) {
				// TODO Auto-generated method stub
				
			}

			@Override
			public void mouseReleased(MouseEvent arg0) {
				// TODO Auto-generated method stub
				
			}
		});
		
		model.addListener(new Listener<Boolean>() {

			@Override
			public void notifyChange(Boolean data) {
				checkBox.setSelected(data);
				enable1();
			}
		});
	}
	/**
	 * Update enabled-ness of panel. Just calling enable() doesn nothing?
	 */
	private void enable1() {
		boolean enabled = !model.isLocked();
		checkBox.setEnabled(enabled);
		label.setEnabled(enabled);
	}

	public JCheckBox getCheckBox() {
		return checkBox;
	}

}