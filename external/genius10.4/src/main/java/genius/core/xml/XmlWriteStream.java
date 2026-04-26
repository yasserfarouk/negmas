package genius.core.xml;

import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

/**
 * a xml stream that can write {@link Map}s
 *
 */
public class XmlWriteStream {

	private final XMLStreamWriter stream;

	/**
	 * 
	 * @param out
	 *            the {@link OutputStream}
	 * @param topLabel
	 *            the top level element name, typically "Tournament" or
	 *            "Session"
	 * @throws XMLStreamException
	 */
	public XmlWriteStream(OutputStream out, String topLabel)
			throws XMLStreamException {
		XMLOutputFactory output = XMLOutputFactory.newInstance();
		stream = output.createXMLStreamWriter(out);

		stream.writeStartDocument();
		stream.writeStartElement(topLabel);
	}

	/**
	 * write data to xml file
	 * 
	 * @param name
	 *            the name of the element to write.
	 * @param data
	 *            a xml map of key-value pairs. The hashmap is written as a full
	 *            element. Then each pair is checked. If the value is a
	 *            {@link Map}, we call write recursively. Otherwise, we convert
	 *            the key and value to {@link String} and write that element as
	 *            an attribute.
	 * 
	 *            To improve layout, the hashmap's string values are written
	 *            first.
	 * 
	 *            If you want to have multiple keys that look identical, use a
	 *            {@link Key} as key.
	 * @throws XMLStreamException
	 */
	@SuppressWarnings("unchecked")
	public void write(String name, Map<Object, Object> data)
			throws XMLStreamException {
		stream.writeCharacters("\n");
		stream.writeStartElement(name);

		for (Object key : data.keySet()) {
			if (!(data.get(key) instanceof Map<?, ?>)) {
				Object value = data.get(key);
				stream.writeAttribute(key.toString(),
						value == null ? "null" : value.toString());
			}
		}

		for (Object key : data.keySet()) {
			if (data.get(key) instanceof Map<?, ?>) {
				write(key.toString(), (HashMap<Object, Object>) data.get(key));
			}
		}
		stream.writeCharacters("\n");
		stream.writeEndElement();
	}

	/**
	 * Close the stream. After this you should dispose the XmlWriteStream as
	 * calls to write and close will fail.
	 */
	public void close() {
		try {
			stream.writeEndDocument();
			stream.flush();
			stream.close();
		} catch (XMLStreamException e) {
			e.printStackTrace();
		}
	}

	public void flush() throws XMLStreamException {
		stream.flush();
	}
}
