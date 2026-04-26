package genius.gui.panels;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import genius.core.listener.Listener;

/**
 * Panel showing a checkbox with a title and value. This panel supports a
 * BooleanModel (you can't with the default {@link Checkbox}).
 */
@SuppressWarnings("serial")
public class ButtonPanel extends JPanel{

	private JButton button;
	private BooleanModel model;

	/**
	 * 
	 * @param text
	 *            the text for the label
	 * @param boolModel
	 *            the boolean model
	 */
	public ButtonPanel(final String text, final BooleanModel boolModel) {
		this.model = boolModel;

		setLayout(new BorderLayout());
		button = new JButton(text);
		enable1();

		// connect the box to the model,

		add(button, BorderLayout.CENTER);
	
		button.addMouseListener(new MouseListener() {

			@Override
			public void mouseClicked(MouseEvent arg0) {
					model.setValue(true);
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
			button.setSelected(data);
			enable1();
		}
	});
}
	
	/**
	 * Update enabled-ness of panel. Just calling enable() doesn nothing?
	 */
	private void enable1() {
		boolean enabled = !model.isLocked();
		button.setEnabled(enabled);
	}

	public JButton getButton() {
		return button;
	}

}
