import asyncio
from pydngconverter import DNGConverter, flags
dirpath = 'D:/TCL/data_collection/20210702PM_ZOO/'

async def main():
    # Create converter instance.
    pydng = DNGConverter(dirpath,
                        dest='D:/TCL/data_collection/20210702PM_ZOO_dngfiles/',
                        jpeg_preview=flags.JPEGPreview.EXTRACT,
                        fast_load=True,
                        )
    # Convert all
    return await pydng.convert()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()