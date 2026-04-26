package negotiator.repository.boa;

import java.io.StringReader;
import java.io.StringWriter;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;

import org.junit.Test;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.repository.boa.BoaRepItem;
import genius.core.repository.boa.BoaWithSettingsRepItem;
import genius.core.repository.boa.ParameterList;

public class BoaWithSettingsRepItemTest {

	@Test
	public void testDeserialize() throws JAXBException {
		JAXBContext jaxbContext = JAXBContext.newInstance(BoaWithSettingsRepItem.class);
		Unmarshaller unmarshaller = jaxbContext.createUnmarshaller();
		String TEXT = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><boawithsettings><parameters/><item classpath=\"class.path\"/></boawithsettings>";
		StringReader reader = new StringReader(TEXT);
		BoaWithSettingsRepItem element = (BoaWithSettingsRepItem) unmarshaller.unmarshal(reader);

		System.out.println(element);
	}

	@Test
	public void testSerialize() throws JAXBException {
		BoaRepItem<AcceptanceStrategy> ac = new BoaRepItem<AcceptanceStrategy>("class.path");
		ParameterList params = new ParameterList();
		BoaWithSettingsRepItem<AcceptanceStrategy> elt = new BoaWithSettingsRepItem<AcceptanceStrategy>(ac, params);
		StringWriter writer = new StringWriter();

		JAXBContext context = JAXBContext.newInstance(BoaWithSettingsRepItem.class);
		Marshaller m = context.createMarshaller();
		m.marshal(elt, writer);
		System.out.println(writer.toString());
	}
}
