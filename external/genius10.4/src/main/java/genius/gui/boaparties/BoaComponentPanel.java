package genius.gui.boaparties;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.BOA;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.BoaType;
import genius.core.exceptions.InstantiateException;
import genius.core.repository.boa.BoaRepItem;
import genius.gui.renderer.RepItemListCellRenderer;

/**
 * Allows user to pick properties of a Boa Component. Contains a row with the
 * component (combobox) and a row with the settings (shows values as text and a
 * "Change" button)..
 * 
 * @param T
 *            the type of the BOA components
 */
@SuppressWarnings("serial")
public class BoaComponentPanel<T extends BOA> extends JPanel {

	public BoaComponentPanel(final BoaComponentModel<T> model, String title) {
		setLayout(new BorderLayout());
		add(new JLabel(title), BorderLayout.NORTH);
		JComboBox<BoaRepItem<T>> combo = new JComboBox<BoaRepItem<T>>(model.getComponentsListModel());
		combo.setRenderer(new RepItemListCellRenderer());
		add(combo, BorderLayout.CENTER);
		add(new BoaSettingsArea(model.getParameters()), BorderLayout.SOUTH);
	}

	/**
	 * Simple test function
	 * 
	 * @param args
	 * @throws InstantiateException
	 * @throws InterruptedException
	 */
	public static void main(String[] args) throws InstantiateException, InterruptedException {
		JFrame frame = new JFrame();
		frame.setLayout(new BorderLayout());

		BoaComponentModel<AcceptanceStrategy> model = new BoaComponentModel<>(BoaType.ACCEPTANCESTRATEGY);

		frame.getContentPane().add(new BoaComponentPanel<AcceptanceStrategy>(model, "test boa component panel"),
				BorderLayout.CENTER);
		frame.pack();
		frame.setVisible(true);
	}

}

/**
 * The lower part of the panel: the settings as text, plus a change button.
 */
@SuppressWarnings("serial")
class BoaSettingsArea extends JPanel {
	private JTextField text = new JTextField();
	private JButton change = new JButton("Change");
	private BoaParametersModel model;

	public BoaSettingsArea(final BoaParametersModel model) {
		this.model = model;
		setLayout(new BorderLayout());
		text.setEditable(false);

		add(text, BorderLayout.CENTER);
		add(change, BorderLayout.EAST);

		// handle change button click
		change.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				BoaParametersPanel parampanel = new BoaParametersPanel(model);
				int result = JOptionPane.showConfirmDialog(BoaSettingsArea.this, parampanel, "enter parameters",
						JOptionPane.OK_CANCEL_OPTION);
				if (result == JOptionPane.OK_OPTION) {
					System.out.println("result=" + model);
				}
			}

		});

		// update when parameters change
		model.addTableModelListener(new TableModelListener() {
			@Override
			public void tableChanged(TableModelEvent e) {
				update();
			}
		});

		update();
	}

	private void update() {
		List<BOAparameter> setting = model.getSetting();
		text.setText(setting.toString());
		change.setEnabled(!setting.isEmpty());
	}

}
