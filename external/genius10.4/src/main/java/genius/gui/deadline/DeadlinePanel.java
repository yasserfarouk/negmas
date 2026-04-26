package genius.gui.deadline;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import genius.core.DeadlineType;
import genius.core.listener.Listener;

/**
 * panel allowing user to set the deadline.
 * 
 * @author W.Pasman
 *
 */
@SuppressWarnings("serial")
public class DeadlinePanel extends JPanel {
	private final SpinnerNumberModel valuemodel = new SpinnerNumberModel(60, 1,
			10000, 10);
	private JSpinner spinner = new JSpinner(valuemodel);
	private JComboBox<DeadlineType> combobox = new JComboBox<DeadlineType>(
			DeadlineType.values());
	private JLabel label = new JLabel("Deadline");
	private DeadlineModel model;

	public DeadlinePanel(final DeadlineModel model) {
		if (model == null)
			throw new NullPointerException("model");
		this.model = model;
		setLayout(new BorderLayout());
		label.setPreferredSize(new Dimension(120, 10));
		add(label, BorderLayout.WEST);
		add(spinner, BorderLayout.CENTER);
		add(combobox, BorderLayout.EAST);
		setMaximumSize(new Dimension(Short.MAX_VALUE, 30));
		setAlignmentX(Component.RIGHT_ALIGNMENT);

		syncWithModel();

		spinner.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				model.setValue((Integer) valuemodel.getValue());
			}
		});
		combobox.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				model.setType((DeadlineType) combobox.getSelectedItem());
			}
		});

		model.addListener(new Listener<DeadlineModel>() {

			@Override
			public void notifyChange(DeadlineModel data) {
				syncWithModel();
			}

		});
	}

	/**
	 * Sync panel with the model.
	 */
	private void syncWithModel() {
		spinner.setValue(model.getDeadline().getValue());
		combobox.setSelectedItem(model.getDeadline().getType());
	}

}
