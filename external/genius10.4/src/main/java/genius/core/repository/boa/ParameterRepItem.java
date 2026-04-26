package genius.core.repository.boa;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

/**
 * Rep item storing a parameter value
 */
@XmlRootElement(name = "param")
public class ParameterRepItem {
	@XmlAttribute
	private String name;
	@XmlAttribute
	private Double value;

	@SuppressWarnings("unused")
	private ParameterRepItem() {
	}

	public ParameterRepItem(String name, Double value) {
		this.name = name;
		this.value = value;
	}

	public String getName() {
		return name;
	}

	public Double getValue() {
		return value;
	}

	@Override
	public String toString() {
		return name + "=" + value;
	}

}
