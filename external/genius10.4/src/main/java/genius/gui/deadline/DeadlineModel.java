package genius.gui.deadline;

import genius.core.Deadline;
import genius.core.DeadlineType;
import genius.core.listener.DefaultListenable;

/**
 * Stores the selected deadline type and can notify about changes.
 */
public class DeadlineModel extends DefaultListenable<DeadlineModel> {

	// default values.
	private Integer value = 60;
	private DeadlineType type = DeadlineType.ROUND;

	public void setValue(Integer value) {
		this.value = value;
		notifyChange(null);
	}

	public void setType(DeadlineType type) {
		this.type = type;
		notifyChange(null);
	}

	/**
	 * @return the current deadline
	 */
	public Deadline getDeadline() {
		return new Deadline(value, type);
	}

}
